from params_proto import cli_parse
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from plan2vec.plotting.maze_world.embedding_image_maze import \
    cache_images, \
    visualize_embedding_2d_image, \
    visualize_embedding_3d_image, visualize_value_map


@cli_parse
class Args:
    seed = 0

    # environment specs
    env_id = 'GoalMassDiscreteImgIdLess-v0'
    goal_key = "goal_img"
    obs_key = "img"
    num_envs = 20
    n_rollouts = 1000
    timesteps = 2

    latent_dim = 2

    pair_distance = 0.05
    optim_sample_limit = 1_000  # 20 neighbors per sample.

    k_fold = 10

    batch_size = 100
    n_workers = 5

    download_mnist = False

    n_epochs = 1000
    lr = 1e-4  # 0.01 for SGD
    optimizer = "Adam"

    use_gpu = True

    checkpoint_last = True


def sample_data(**kwargs):
    import numpy as np
    from tqdm import trange
    from ml_logger import logger
    from plan2vec.mdp.replay_buffer import ReplayBuffer
    from plan2vec.mdp.wrappers.subproc_vec_env import SubprocVecEnv
    from plan2vec.mdp.helpers import make_env
    from plan2vec.mdp.sampler import path_gen_fn

    Args.update(**kwargs)

    np.random.seed(Args.seed)

    envs = SubprocVecEnv([make_env(Args.env_id, Args.seed + i) for i in range(Args.num_envs)])
    logger.log_params(env=envs.spec._kwargs)

    memory = ReplayBuffer(Args.n_rollouts * Args.timesteps)

    random_pi = lambda ob, goal, *_: np.random.randint(0, 8, size=[len(ob)])
    random_path_gen = path_gen_fn(envs, random_pi, Args.obs_key, Args.goal_key,
                                  all_keys=['x', 'goal'] + [Args.obs_key, Args.goal_key])
    next(random_path_gen)

    img_size = 64  # hard coded by env.

    for i in trange(Args.n_rollouts // Args.num_envs, desc=f"sampling from {Args.env_id}"):
        paths = random_path_gen.send(Args.timesteps)
        memory.extend(obs=paths['obs'][Args.obs_key].reshape(-1, 1, img_size, img_size),
                      s=paths['obs']['x'].reshape(-1, 2))
    envs.close()

    return memory['obs'], memory['s']


import numpy as np


class PairDataset(Dataset):
    def __init__(self, all_images, inds_1, inds_2, labels, dtype=torch.float, device=None):
        self.all_images = all_images
        self.inds_1 = inds_1
        self.inds_2 = inds_2
        self.labels = labels
        self.dtype = dtype
        self.device = device

    def __getitem__(self, index):
        if isinstance(index, int):
            return torch.tensor(self.all_images[self.inds_1[index]], dtype=self.dtype, device=self.device), \
                   torch.tensor(self.all_images[self.inds_2[index]], dtype=self.dtype, device=self.device), \
                   torch.tensor(self.labels[index], dtype=self.dtype, device=self.device)

        elif isinstance(index, slice):
            data_len = len(self.all_images)
            # this can potentially introduce a memory leak.
            start = np.floor(data_len * index.start).astype(int) if isinstance(index.start, float) else index.start
            stop = np.floor(data_len * index.stop).astype(int) if isinstance(index.stop, float) else index.stop
            return PairDataset(all_images=self.all_images,
                               inds_1=self.inds_1[start:stop],
                               inds_2=self.inds_2[start:stop],
                               labels=self.labels[start:stop],
                               dtype=self.dtype,
                               device=self.device)
        else:
            raise KeyError(f"{index} slice is not supported!")

    def __len__(self):
        return len(self.inds_1)


# 1. randomly sample pairs, supervise with distance
# 2. sample pairs, plot MSE over grount-truth distance. Show increased MSE
# 3. train over smaller, evaluate over, show MSE.

def train(**kwargs):
    import numpy as np
    import torch.nn.functional as F
    from ml_logger import logger

    Args.update(**kwargs)
    device = torch.device("cuda" if Args.use_gpu and torch.cuda.is_available() else "cpu")
    logger.log_params(Args=vars(Args))

    np.random.seed(Args.seed)
    torch.manual_seed(Args.seed)
    # see https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    all_images, all_states = sample_data()

    max_distance = np.linalg.norm(all_states - all_states[::-1], ord=2, axis=-1).max(axis=0)
    logger.log_line(f"Maximum distance is {max_distance}.\n"
                    f"Pair range is {Args.pair_distance}.\n"
                    f"Datasize is {len(all_states)}", file="README.md")

    def get_pairs(pair_d):
        logger.log("computing the pair-wise distances...", color="yellow")
        distances = np.linalg.norm(all_states[None, ...] - all_states[:, None, ...], ord=2, axis=-1)
        logger.log("finished...", color="green")
        inds = distances < pair_d
        _ = np.arange(len(all_states))
        ijs = np.meshgrid(_, _)
        a, b = ijs[0][inds], ijs[1][inds]
        return a, b, all_states[a], all_states[b]

    inds_1, inds_2, xys_1, xys_2 = get_pairs(Args.pair_distance)

    from plan2vec.models.convnets import LocalMetricConvLarge

    # use 2x of the latent_dim for variance
    model = LocalMetricConvLarge(1, Args.latent_dim)
    logger.log_text(str(model), "model/ConvNet.txt")
    logger.log_line(*[f"{k}: {v.shape}" for k, v in model.state_dict().items()], sep="\n",
                    file="model/model_details.txt")
    model.to(device)

    optimizer = getattr(torch.optim, Args.optimizer)(model.parameters(), Args.lr)

    dataset = PairDataset(all_images, inds_1, inds_2, labels=np.linalg.norm(xys_1 - xys_2, ord=2, axis=-1),
                          dtype=torch.float, device=device)

    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn')

    train_loader = DataLoader(dataset[1 / Args.k_fold:], batch_size=Args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset[:1 / Args.k_fold], batch_size=Args.batch_size, shuffle=True)

    def compute_error():
        with torch.no_grad():
            model.eval()
            for i, (img_1, img_2, d) in enumerate(tqdm(test_loader, desc="evaluating...")):
                if Args.optim_sample_limit and i > Args.optim_sample_limit:
                    break
                d_hat = model(img_1, img_2).squeeze(-1)
                loss = MSE_criterion(d_hat, d)
                rms = torch.sqrt((d_hat - d).pow(2).mean())
                logger.store_metrics(metrics={
                    "eval/mse": loss.item(),
                    "eval/rms": rms.item(),
                    "eval/error": torch.abs(d_hat - d).cpu().numpy()
                })
        model.train(True)

    MSE_criterion = torch.nn.MSELoss(size_average=None, reduction='mean')
    for epoch in range(Args.n_epochs):
        for i, (img_1, img_2, d) in enumerate(tqdm(train_loader, desc=f"training@epoch{epoch}")):
            if Args.optim_sample_limit and i > Args.optim_sample_limit:
                break
            d_hat = model(img_1, img_2).squeeze(-1)
            loss = MSE_criterion(d_hat, d)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                logger.store_metrics(metrics={
                    "train/loss": loss.item(),
                    "train/error": torch.abs(d_hat - d).detach().cpu().numpy()
                })

        compute_error()
        logger.log_metrics_summary(default_stats="mean",
                                   key_stats={"eval/error": "quantile", "train/error": "quantile"},
                                   key_values=dict(epoch=epoch, dt_epoch=logger.split()))

    if Args.checkpoint_last:
        logger.save_module(model, "checkpoints/encoder.pkl", 10_000_000)

    logger.print('training is complete', color="green")


if __name__ == "__main__":
    train()
