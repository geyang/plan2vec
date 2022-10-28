import torch
from params_proto.neo_proto import ParamsProto, Proto
from torch import optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import trange

from plan2vec.mdp.replay_buffer import ReplayBuffer
from plan2vec.plotting.visualize_traj_2d import visualize_prediction, local_metric_vs_ground_truth, \
    visualize_eval_prediction
from torch_utils import RMSLoss

from plan2vec.models.convnets import LocalMetricConvLarge, LocalMetricConvLargeKernel, LocalMetricConvLargeL2
from plan2vec.models.resnet import ResNet18CoordL2

assert [LocalMetricConvLarge, LocalMetricConvLargeKernel, LocalMetricConvLargeL2,
        ResNet18CoordL2], "some used in analysis."


class TripletLossDataset(Dataset):
    def __init__(self, X, X_lo, Y):
        self.X = X
        self.X_lo = X_lo
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.X_lo[idx], self.Y[idx]


class Args(ParamsProto):
    seed = 0

    env_id = 'GoalMassDiscreteImgIdLess-v0'
    goal_key = "goal_img"
    obs_key = "img"
    num_envs = 20
    n_rollouts = 1000
    timesteps = 10

    local_metric = "LocalMetricConvLarge"
    load_weights = Proto(None, help="path for trained weights", dtype=str)
    save_weights = Proto("models/local_metric.pkl", help="location to save the weight")

    latent_dim = 8

    num_epochs = 20
    batch_size = 32
    # lr = LinearAnneal(1e-3, min=1e-3, n=num_epochs + 1)
    lr = 1e-4
    k_fold = Proto(10, dtype=int, help="The k-fold for validation")

    vis_interval = 10


def train(memory: ReplayBuffer, metric_fn, random_path_gen):
    """Build dataset and train local metric for passive setting."""
    from ml_logger import logger

    batch = memory.sample(len(memory))
    zero_labels = np.zeros(len(memory))
    pos_labels = np.ones(len(memory))
    neg_labels = np.ones(len(memory)) * 2

    # todo: consider supervising with actual distance.
    shuffle = np.random.rand(len(batch['s_'])).argsort()
    x = np.concatenate([batch['obs'], batch['obs'], batch['obs']])
    x_prime = np.concatenate([batch['obs'], batch['obs_'], batch['obs_'][shuffle]])
    s = np.concatenate([batch['s'], batch['s'], batch['s']])
    s_prime = np.concatenate([batch['s'], batch['s_'], batch['s_'][shuffle]])
    all_x = np.array(list(zip(x, x_prime)))
    all_s = np.array(list(zip(s, s_prime)))
    all_y = np.concatenate([zero_labels, pos_labels, neg_labels])
    # visualize_prediction(s, s_prime, all_y, k=20, key="figures/sample.png", title="Training Set")

    shuffle = np.random.rand(len(all_x)).argsort()
    valid_index = len(all_x) // Args.k_fold
    loader = DataLoader(TripletLossDataset(all_x[shuffle][:valid_index], all_s[shuffle][:valid_index],
                                           all_y[shuffle][:valid_index]),
                        batch_size=Args.batch_size, shuffle=True, num_workers=4)
    eval_loader = DataLoader(TripletLossDataset(all_x[shuffle][valid_index:], all_s[shuffle][valid_index:],
                                                all_y[shuffle][valid_index:]),
                             batch_size=valid_index, shuffle=True, num_workers=4)

    optimizer = optim.Adam(metric_fn.parameters(), lr=Args.lr)

    rms = RMSLoss()

    def evaluate():
        with torch.no_grad():
            for i, (_, s, y) in enumerate(eval_loader):
                x, x_prime = _[:, 0, :], _[:, 1, :]
                y_hat = metric_fn(x.float().to(Args.device), x_prime.float().to(Args.device)).squeeze(-1)
                correct = lambda th: (y_hat.cpu() < th).numpy() == (y.cpu().numpy() < th)
                logger.store_metrics(metrics={f"eval/accuracy-{th}": correct(th) for th in [0.5, 1.1, 1.5]},
                                     y=y.mean(), sample_bias=y.numpy())

                s, s_prime = s[:, 0, :], s[:, 1, :]

            local_metric_vs_ground_truth(s.cpu().numpy(),
                                         s_prime.cpu().numpy(),
                                         y_hat.cpu().numpy(),
                                         yDomain=[0, 3],
                                         key=f"figures/score_vs_l1_{epoch:04d}.png")

    for epoch in range(Args.num_epochs + 1):
        for i, (_, s, y) in enumerate(loader):
            x, x_prime = _[:, 0, :], _[:, 1, :]
            y_hat = metric_fn(x.float().to(Args.device), x_prime.float().to(Args.device)).squeeze(-1)
            # y_hat, mu, logvar = metric_fn(x.float().to(Args.device), x_prime.float().to(Args.device))
            loss = F.smooth_l1_loss(y_hat.view(-1), y.view(-1).float().to(Args.device))
            # loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            logger.store_metrics(loss=loss.cpu().item())

            # if epoch % Args.vis_interval == 0:
            #     visualize_pairs(all_x[:valid_index][0].numpy(), s_prime.cpu().numpy(), f"figures/transitions_{epoch:04d}.png")

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct = lambda th: (y_hat.cpu() < th).numpy() == (y.cpu().numpy() < th)
            with torch.no_grad():
                logger.store_metrics(metrics={f"accuracy-{th}": correct(th) for th in [0.5, 1.1, 1.5]},
                                     loss=loss.cpu().item(), y_hat=y_hat.mean().cpu().item(), y=y.mean().cpu().item(), )

        evaluate()

        logger.log_metrics_summary(key_values=dict(epoch=epoch), default_stats='mean')

    if Args.save_weights:
        with logger.SyncContext():  # to make sure that the weight files are saved completely.
            logger.save_module(metric_fn, "models/local_metric.pkl", chunk=200_000_000, show_progress=True)


def main(deps, freeze_after=True, **kwargs):
    from ml_logger import logger
    from plan2vec.mdp.wrappers.subproc_vec_env import SubprocVecEnv
    from plan2vec.mdp.helpers import make_env
    from plan2vec.mdp.sampler import path_gen_fn

    Args._update(deps, **kwargs)
    Args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f_local_metric = globals()[Args.local_metric](1, Args.latent_dim).to(Args.device)
    Args.model = type(f_local_metric).__name__
    logger.log_params(Args=vars(Args))
    logger.log_text(str(f_local_metric), filename="models/f_local_metric.txt")

    np.random.seed(Args.seed)
    torch.manual_seed(Args.seed)
    # see https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    envs = SubprocVecEnv([make_env(Args.env_id, Args.seed + i) for i in range(Args.num_envs)])
    logger.log_params(env=envs.spec._kwargs)

    memory = ReplayBuffer(Args.n_rollouts * Args.timesteps)

    random_pi = lambda ob, goal, *_: np.random.randint(0, 8, size=[len(ob)])
    random_path_gen = path_gen_fn(envs, random_pi, Args.obs_key, Args.goal_key,
                                  all_keys=['x', 'goal'] + [Args.obs_key, Args.goal_key])
    next(random_path_gen)
    img_size = 64

    for i in trange(Args.n_rollouts // Args.num_envs, desc=f"sampling from {Args.env_id}"):
        paths = random_path_gen.send(Args.timesteps)
        memory.extend(obs=paths['obs'][Args.obs_key].reshape(-1, 1, img_size, img_size),
                      obs_=paths['next'][Args.obs_key].reshape(-1, 1, img_size, img_size),
                      s=paths['obs']['x'].reshape(-1, 2),
                      s_=paths['next']['x'].reshape(-1, 2))

    train(memory, f_local_metric, random_path_gen)

    if freeze_after:
        for p in f_local_metric.parameters():
            p.requires_grad = False

    return f_local_metric.eval()


if __name__ == "__main__":
    from ml_logger import logger
    from plan2vec_experiments import instr

    _ = instr(main, seed=5 * 100)
    if True:
        with logger.SyncContext():
            _()
    elif True:
        cprint('Training on cluster', 'green')
        import jaynes

        jaynes.config("devfair")
        jaynes.run(_, seed=5 * 100)
        jaynes.listen()
