"""
Rope Dataset Local Metric, with negative examples from within the trajectory.
"""
from copy import copy

import torch
from termcolor import cprint
from torch import optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from params_proto.neo_proto import ParamsProto, Proto
from tqdm import tqdm, trange

from plan2vec.data_utils.rope_dataset import SeqDataset
from plan2vec.models.convnets import LocalMetricConvLarge, LocalMetricConvDeep
from plan2vec.plotting.rope_viz import score_distribution, top_neighbors, faraway_samples

# to fix dataset loading erorr
torch.multiprocessing.set_sharing_strategy('file_system')


def normalize(x):
    """Normalizes the

    :return: torchTensor(dtype=float64)
    """
    return (x - x.min()) / (x.max() - x.min())


class PairDataset(Dataset):
    """
    Pair Dataset with in-trajectory negative examples from within the trajectory.
    """

    def __init__(self, data, K, subsample=1, rest_subsample=10, shuffle=True):
        """ Creates a Paired dataset object

        :param subsample: sub-sample the trajectory data.
        :param data: numpy list of trajectory tensors
        :param shuffle: boolean flag for whether shuffle the order of different trajectories
        """
        self.K = K
        if shuffle:  # shuffles by trajectory.
            data = copy(data)
            np.random.shuffle(data)

        self.data = [traj.transpose(0, 3, 1, 2).astype(np.float32) / 255 for traj in data]

        inds, inds_prime, labels = [], [], []
        for i, traj in enumerate(data):  # Iterate through each trajectory
            traj = traj.astype(np.float32)
            for k in [*range(0, K, subsample), *range(K, len(traj), rest_subsample)]:
                for j in range(len(traj) - k):
                    inds.append([i, j])
                    inds_prime.append([i, j + k])
                    # labels.append((k / K) if k <= K else 1)
                    labels.append(0 if k <= K else 1)

        # add in-trajectory negative samples
        # change labels to 0, 1, 2, k, 100
        shuffled_inds = np.random.rand(len(inds)).argsort() % len(inds)
        self.indices = np.concatenate([inds, inds_prime, inds])
        self.indices_prime = np.concatenate([inds_prime, inds, np.array(inds_prime)[shuffled_inds]])
        self.labels = np.concatenate([labels, labels, 1 * np.ones_like(shuffled_inds)]).astype(np.float32)

    def __getitem__(self, index):
        i, j = self.indices[index]
        i_, j_ = self.indices_prime[index]
        if self.labels[index] < 2:
            return (
                       self.data[i][j], self.data[i_][j_]
                   ), self.labels[index]
        else:
            i_ = np.random.randint(0, len(self.data))
            j_ = np.random.randint(0, len(self.data[i_]))
            while i_ == i and abs(j - j_) <= self.K:
                j_ = np.random.randint(0, len(self.data[i_]))
            return (self.data[i][j], self.data[i_][j_]), np.float32(2)

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return f'PairDataset("")'


class Args(ParamsProto):
    env_id = "rope"

    data_path = Proto("~/fair/new_rope_dataset/data/new_rope.npy", help="path to the data file")
    seed = 0
    load_weights = Proto(None, help="path for trained weights", dtype=str)
    save_weights = Proto("models/local_metric.pkl", help="location to save the weight")

    latent_dim = 32

    num_epochs = 100
    batch_size = 16
    # lr = LinearAnneal(1e-3, min=1e-3, n=num_epochs + 1)
    lr = 1e-4
    beta_1 = 0.9
    beta_2 = 0.999
    weight_decay = Proto(0.01, help="should be close to zero.")
    k_fold = Proto(10, dtype=int, help="The k-fold for validation")

    # sample bias
    K = Proto(2, help="The regularizing constant for how far we consider neighbors.")
    subsample = Proto(1, help="sub-sample the trajectory")
    rest_subsample = Proto(10, help="sub-sample the trajectory")

    vis_k = Proto(10, help="The number of sample grid we show per epoch")
    vis_interval = 10
    vis_horizon = Proto(50, help="the horizon for visualizing score vs timesteps")


class DEBUG(ParamsProto):
    show_sample_trajectories = Proto(False, help="debug flag for saving sample trajectories")


def train(metric_fn):
    """Build dataset and train local metric for passive setting."""
    from ml_logger import logger
    from os.path import expanduser

    rope = np.load(expanduser(Args.data_path), allow_pickle=True)
    l = 0 if Args.k_fold is None else len(rope) // Args.k_fold
    dataset = PairDataset(rope[l:], K=Args.K, subsample=Args.subsample,
                          rest_subsample=Args.rest_subsample)
    loader = DataLoader(dataset, batch_size=Args.batch_size, shuffle=True, num_workers=0)
    score_distribution(dataset.labels, f"figures/train_scores.png")
    if Args.k_fold is not None:
        eval_dataset = PairDataset(rope[:len(rope) // Args.k_fold], K=Args.K, subsample=Args.subsample,
                                   rest_subsample=Args.rest_subsample)
        eval_loader = DataLoader(eval_dataset, batch_size=Args.batch_size, shuffle=True, num_workers=0)
        score_distribution(eval_dataset.labels, f"figures/eval_scores.png")

    # used for the score distribution stats. Make smaller for faster training
    all_images = torch.tensor(np.concatenate(rope).transpose(0, 3, 1, 2)[::1] / 255, dtype=torch.float32,
                              device=Args.device, requires_grad=False)
    traj_labels = torch.tensor(np.concatenate([np.ones(len(traj)) * i for i, traj in enumerate(rope)])[::1],
                               device=Args.device, requires_grad=False)

    # used for visualization. Deterministic
    seq_gen = DataLoader(SeqDataset(rope, H=Args.vis_horizon, shuffle=False), batch_size=20, shuffle=True,
                         num_workers=0)
    eval_trajs = next(iter(seq_gen)).to(Args.device)
    eval_x = eval_trajs[:, :1, :, :, :]

    if DEBUG.show_sample_trajectories:
        cprint(f'saving evaluation trajectory data: {eval_trajs.shape}', "yellow")
        for i in trange(eval_trajs.shape[1]):
            logger.log_images(eval_trajs[:, i].cpu().numpy().transpose(0, 2, 3, 1),
                              f"figures/eval_trajs/step_{i:04d}.png", n_rows=4, n_cols=5)

    optimizer = optim.Adam(metric_fn.parameters(), lr=Args.lr, betas=[Args.beta_1, Args.beta_2],
                           weight_decay=Args.weight_decay)
    criteria = torch.nn.BCEWithLogitsLoss()

    # from torch_utils import RMSLoss
    # rms = RMSLoss()
    def evaluate(epoch, step):
        nonlocal all_images, traj_labels
        prefix = f"figures/epoch_{epoch:04d}-{step:04d}"
        if Args.k_fold is not None:
            for i, ((s, s_prime), y) in enumerate(tqdm(eval_loader, total=len(eval_loader.dataset) // Args.batch_size)):
                if i < Args.vis_k and (epoch % Args.vis_interval == 0 or epoch < Args.vis_interval):
                    s.requires_grad_(True)
                    s_prime.requires_grad_(True)

                    y_hat = metric_fn(s.to(Args.device), s_prime.to(Args.device)).squeeze()
                    # loss = F.smooth_l1_loss(y_hat, y.view(-1).to(Args.device))
                    loss = criteria(y_hat, y.view(-1).to(Args.device))
                    loss.backward()

                with torch.no_grad():
                    y_hat = metric_fn(s.to(Args.device), s_prime.to(Args.device)).squeeze()
                correct = lambda th: (y_hat.cpu() < th).numpy() == (y.cpu().numpy() < th)
                logger.store_metrics(
                    metrics={f"eval/accuracy-{th}": correct(th) for th in [0.1, 0.2, 0.4, 0.5, 0.6, 0.8]},
                    y=y.mean(), sample_bias=y.numpy())

        # score vs timesteps
        with torch.no_grad():
            y_hat = metric_fn(eval_x, eval_trajs).squeeze()
            from plan2vec.plotting.visualize_traj_2d import local_metric_over_trajectory
            local_metric_over_trajectory(y_hat.cpu(), f"{prefix}/score_vs_timesteps_{epoch:04d}.png", ylim=(-0.1, 1.1),
                                         xlim=(-0.1, 10.1))
            score_distribution(y_hat.cpu(), f"{prefix}/in_trajectory_scores.png", xlim=[-.1, 0.2])
            score_distribution(y_hat.cpu(), f"{prefix}/in_trajectory_scores_full.png", xlim=[-0.1, 1.2])

            # y_hat = metric_fn(all_images[:, None, :, :, :], all_images[None, :1, :, :, :]).squeeze()
            # score_distribution(y_hat.cpu(), f"{prefix}/all_scores.png", xlim=[-.1, 0.2])
            # score_distribution(y_hat.cpu(), f"{prefix}/all_scores_full.png", xlim=[-0.1, 1.2])
            # for _ in range(0, 1000, 100):
            #     y_hat = metric_fn(all_images[:, None, :, :, :],
            #                       all_images[None, _: _ + 1, :, :, :]).squeeze()
            #     top_neighbors(all_images, all_images[_, :, :, :], y_hat, f"{prefix}/top_neighbors_{_:04d}.png")
            #     faraway_samples(all_images, all_images[_, :, :, :], y_hat, f"{prefix}/faraway_samples_{_:04d}.png")

    num_batches = len(loader.dataset) // Args.batch_size

    logger.split()
    for epoch in range(Args.num_epochs + 1):

        for i, ((s, s_prime), y) in enumerate(tqdm(loader, total=num_batches, desc=f"epoch: {epoch}")):
            y_hat = metric_fn(s.to(Args.device), s_prime.to(Args.device))
            loss = F.smooth_l1_loss(y_hat.view(-1), y.view(-1).to(Args.device))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct = lambda th: (y_hat.cpu() < th).numpy() == (y.cpu().numpy() < th)
            with torch.no_grad():
                logger.store_metrics(metrics={f"accuracy-{th}": correct(th) for th in [0.1, 0.2, 0.4, 0.5, 0.6, 0.8]},
                                     loss=loss.cpu().item(), y_hat=y_hat.mean().cpu().item(), y=y.mean().cpu().item(), )

            if i % 5000 == 0:
                evaluate(epoch, step=i // 5000)

                if Args.save_weights:
                    logger.save_module(metric_fn, f"models/local_metric_{epoch:03d}-{i // 1000:03d}.pkl",
                                       show_progress=True, chunk=1_000_000)

            if i % 100 == 0:
                logger.log_metrics_summary(default_stat="quantile", key_values=dict(epoch=epoch + i / num_batches, ))

            # free up the data
            del s
            del s_prime

        logger.save_module(metric_fn, f"models/local_metric_{epoch:03d}-{i // 1000:03d}.pkl",
                           show_progress=True, chunk=1_000_000)


def main(freeze_after=True, **kwargs):
    from ml_logger import logger

    Args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Args.device = torch.device("cpu")
    Args._update(kwargs)

    np.random.seed(Args.seed)
    torch.manual_seed(Args.seed)
    # see https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    f_local_metric = LocalMetricConvDeep(1, Args.latent_dim).to(Args.device)
    Args.model = type(f_local_metric).__name__

    logger.log_params(Args=vars(Args))

    if Args.load_weights:
        logger.load_module(f_local_metric, Args.load_weights)

    train(f_local_metric)

    if freeze_after:
        for p in f_local_metric.parameters():
            p.requires_grad = False

    return f_local_metric.eval()


if __name__ == "__main__":
    from ml_logger import logger
    from plan2vec_experiments import instr

    if True:
        instr(main, seed=5 * 100)()

    else:
        cprint('Training on cluster', 'green')
        import jaynes

        jaynes.config("dev-gpu")
        # jaynes.config("devfair")

        for Args.lr in [3e-4]:
            _ = instr(main, seed=5 * 100, lr=Args.lr)
            jaynes.run(_)
        jaynes.listen()
