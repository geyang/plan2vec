from copy import copy

import torch
from torch import optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from params_proto import cli_parse, Proto
from tqdm import tqdm

from plan2vec.data_utils.rope_dataset import SeqDataset
from plan2vec.models.convnets import LocalMetricConvLarge
from plan2vec.plotting.rope_viz import score_distribution, top_neighbors, faraway_samples


def normalize(x):
    """Normalizes the

    :return: torchTensor(dtype=float64)
    """
    return (x - x.min()) / (x.max() - x.min())


class PairDataset(Dataset):
    """ Pair Dataset: supports fractional indices for k-fold validation. """

    def __init__(self, data, K, shuffle=True):
        """ Creates a Paired dataset object

        :param data: numpy list of trajectory tensors
        :param shuffle: boolean flag for whether shuffle the order of different trajectories
        """
        if shuffle:  # shuffles by trajectory.
            data = copy(data)
            np.random.shuffle(data)

        self.data = [traj.transpose(0, 3, 1, 2).astype(np.float32) / 255 for traj in data]

        inds, inds_prime, labels = [], [], []
        for i, traj in enumerate(data):  # Iterate through each trajectory
            traj = traj.astype(np.float32)
            for k in range(0, K):
                for j in range(len(traj) - k):
                    inds.append([i, j])
                    inds_prime.append([i, j + k])
                    labels.append(k / K)

        # add in-trajectory samples
        # change labels to 0, 1, 2, k, 100
        shuffled_inds = np.random.rand(len(inds)).argsort() % len(inds)
        self.indices = np.concatenate([inds, np.array(inds)[shuffled_inds]])
        self.indices_prime = np.concatenate([inds_prime, np.array(inds_prime)[shuffled_inds]])
        self.labels = np.concatenate([labels, 1.1 * np.ones_like(shuffled_inds)]).astype(np.float32)

    def __getitem__(self, index):
        i, j = self.indices[index]
        i_, j_ = self.indices_prime[index]
        return (
                   self.data[i][j], self.data[i_][j_]
               ), self.labels[index]

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return f'PairDataset("")'


@cli_parse
class Args:
    data_path = Proto("~/fair/new_rope_dataset/data/new_rope.npy", help="path to the data file")
    seed = 0
    load_weights = Proto(None, help="path for trained weights", dtype=str)
    save_weights = Proto("models/local_metric.pkl", help="location to save the weight")

    latent_dim = 32

    num_epochs = 10
    batch_size = 32
    # lr = LinearAnneal(1e-3, min=1e-3, n=num_epochs + 1)
    lr = 1e-4
    k_fold = Proto(10, dtype=int, help="The k-fold for validation")

    # sample bias
    K = Proto(10, help="The regularizing constant for how far we consider neighbors.")

    vis_k = Proto(10, help="The number of sample grid we show per epoch")
    vis_interval = 10
    vis_horizon = Proto(500, help="the horizon for visualizing score vs timesteps")


def train(metric_fn):
    """Build dataset and train local metric for passive setting."""
    from ml_logger import logger
    from os.path import expanduser

    rope = np.load(expanduser(Args.data_path))
    dataset = PairDataset(rope[len(rope) // Args.k_fold:], K=Args.K)
    loader = DataLoader(dataset, batch_size=Args.batch_size, shuffle=True, num_workers=4)
    eval_dataset = PairDataset(rope[:len(rope) // Args.k_fold], K=Args.K)
    eval_loader = DataLoader(eval_dataset, batch_size=Args.batch_size, shuffle=True, num_workers=4)
    score_distribution(dataset.labels, f"figures/train_scores.png")
    score_distribution(eval_dataset.labels, f"figures/eval_scores.png")

    # used for the score distribution stats. Make smaller for faster training
    all_images = torch.tensor(np.concatenate(rope).transpose(0, 3, 1, 2)[::10] / 255, dtype=torch.float32,
                              device=Args.device, requires_grad=False)
    traj_labels = torch.tensor(np.concatenate([np.ones(len(traj)) * i for i, traj in enumerate(rope)])[::10],
                               device=Args.device, requires_grad=False)

    # used for visualization. Deterministic
    seq_gen = DataLoader(SeqDataset(rope, H=Args.vis_horizon, shuffle=False), batch_size=20, shuffle=False,
                         num_workers=4)
    eval_trajs = next(iter(seq_gen)).to(Args.device)
    eval_x = eval_trajs[:, :1, :, :, :]

    optimizer = optim.Adam(metric_fn.parameters(), lr=Args.lr)

    # from torch_utils import RMSLoss
    # rms = RMSLoss()
    def evaluate(epoch, step):
        nonlocal seq_gen, all_images, traj_labels
        prefix = f"figures/epoch_{epoch:04d}-{step:04d}"
        for i, ((s, s_prime), y) in enumerate(tqdm(eval_loader), total=len(eval_loader.dataset) // Args.batch_size):
            if i < Args.vis_k and (epoch % Args.vis_interval == 0 or epoch < Args.vis_interval):
                s.requires_grad_(True)
                s_prime.requires_grad_(True)

                y_hat = metric_fn(s.to(Args.device), s_prime.to(Args.device)).squeeze()
                loss = F.smooth_l1_loss(y_hat, y.view(-1).to(Args.device))
                loss.backward()

                # diff = normalize(s_prime - s)
                # diff[:, :, :10, :10] = y[:, None, None, None]
                # diff[:, :, :10, 10:20] = y_hat[:, None, None, None]
                # _ = torch.cat([s, s_prime, diff,  # diff image
                #                normalize(s.grad),  # activation of first image
                #                normalize(s_prime.grad)], dim=1)
                # stack = _.reshape(-1, *_.shape[2:])[:50].detach().cpu().numpy()
                # logger.log_images(stack,
                #                   f"figures/eval_pairs/epoch_{epoch:04d}-{step:04d}/activation_{i:04d}.png", 5, 10)

            # visualize activation and samples
            # with torch.no_grad():
            #     _ = torch.cat([s, s_prime, normalize(s_prime - s)], dim=1)
            #     stack = _.reshape(-1, *_.shape[2:])[:30].cpu().numpy()
            #     if i < Args.vis_k and epoch == 0:
            #         logger.log_images(stack, f"figures/sample_pairs/{i:02d}.png", 5, 6)

            with torch.no_grad():
                y_hat = metric_fn(s.to(Args.device), s_prime.to(Args.device)).squeeze()
            correct = (y_hat.cpu() > 1).numpy() == (y.byte().cpu().numpy() > 1)
            logger.store_metrics(metrics={"eval/accuracy": correct}, y=y.mean(), sample_bias=y.numpy())

        # score vs timesteps
        with torch.no_grad():
            y_hat = metric_fn(eval_x, eval_trajs).squeeze()
            from plan2vec.plotting.visualize_traj_2d import local_metric_over_trajectory
            local_metric_over_trajectory(y_hat.cpu(), f"{prefix}/score_vs_timesteps_{epoch:04d}.png", ylim=(-0.1, 1.1))
            score_distribution(y_hat.cpu(), f"{prefix}/in_trajectory_scores.png", xlim=[-.1, 0.2])
            score_distribution(y_hat.cpu(), f"{prefix}/in_trajectory_scores_full.png", xlim=[-0.1, 1.2])

            y_hat = metric_fn(all_images[:, None, :, :, :], all_images[None, :1, :, :, :]).squeeze()
            score_distribution(y_hat.cpu(), f"{prefix}/all_scores.png", xlim=[-.1, 0.2])
            score_distribution(y_hat.cpu(), f"{prefix}/all_scores_full.png", xlim=[-0.1, 1.2])
            for _ in range(0, 10):
                y_hat = metric_fn(all_images[:, None, :, :, :], all_images[None, _ * 10: _ * 10 + 1, :, :, :]).squeeze()
                top_neighbors(all_images, all_images[_ * 100, :, :, :], y_hat, f"{prefix}/top_neighbors_{_:04d}.png")
                faraway_samples(all_images, all_images[_ * 100, :, :, :], y_hat,
                                f"{prefix}/faraway_samples_{_:04d}.png")

    logger.split()
    for epoch in range(Args.num_epochs + 1):

        for i, ((s, s_prime), y) in enumerate(tqdm(loader), total=len(loader.dataset) // Args.batch_size):
            y_hat = metric_fn(s.to(Args.device), s_prime.to(Args.device))
            loss = F.smooth_l1_loss(y_hat.view(-1), y.view(-1).to(Args.device))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                logger.store_metrics(loss=loss.cpu().item(),
                                     y_hat=y_hat.mean().cpu().item(),
                                     y=y.mean().cpu().item(),
                                     accuracy=(y_hat.cpu() > 1).numpy() == (y.byte().cpu().numpy() > 1))

            if i % 1000 == 0:
                evaluate(epoch, step=i // 1000)

            if i % 100 == 0:
                logger.log_metrics_summary(default_stat="quantile",
                                           key_values=dict(epoch=epoch + i / len(loader.dataset) * Args.batch_size), )

    if Args.save_weights:
        logger.save_module(metric_fn, "models/local_metric.pkl")


def main(freeze_after=True, **kwargs):
    import os
    from ml_logger import logger

    Args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Args.device = torch.device("cpu")
    Args.update(kwargs)

    logger.log_params(local_metric=vars(Args))

    np.random.seed(Args.seed)
    torch.manual_seed(Args.seed)

    f_local_metric = LocalMetricConvLarge(1, Args.latent_dim).to(Args.device)

    if Args.load_weights:
        logger.load_module(f_local_metric, os.path.join(Args.load_weights, "models/local_metric.pkl"))

    train(f_local_metric)

    if freeze_after:
        for p in f_local_metric.parameters():
            p.requires_grad = False

    return f_local_metric.eval()


if __name__ == "__main__":
    from ml_logger import logger
    from plan2vec_experiments import instr

    _ = instr(main, seed=5 * 100, __postfix="local-metric-experiments")

    if True:
        _()
    elif False:
        cprint('Training on cluster', 'green')
        import jaynes

        # jaynes.config("devfair")
        jaynes.config("learnfair-gpu")
        # jaynes.config("dev-gpu")
        jaynes.run(_)
        jaynes.listen()

