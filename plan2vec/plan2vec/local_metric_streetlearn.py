#!/usr/bin/env python
"""
StreetLearn Local Metric, only out-of-trajectory negative examples

Gray-scale, (64 x 64)
"""
from copy import copy

import torch
from termcolor import cprint
from torch import optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from params_proto.neo_proto import ParamsProto, Proto
from tqdm import tqdm

from plan2vec.models.convnets import LocalMetricConvDeep
from plan2vec.models.resnet import ResNet18L2, ResNet18Stacked

assert [ResNet18L2, ResNet18Stacked, LocalMetricConvDeep], "networks need to be loaded"

from plan2vec.plotting.rope_viz import score_distribution


def normalize(x):
    """Normalizes the

    :return: torchTensor(dtype=float64)
    """
    return (x - x.min()) / (x.max() - x.min())


class PairDataset(Dataset):
    """
    Pair Dataset with in-trajectory negative examples from within the trajectory.
    """

    def __init__(self, data, shuffle=True):
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
            l = len(traj)
            for j in range(l):
                inds.append([i, j])
                inds_prime.append([i, j])
                labels.append(0)

            for j in range(l - 1):
                inds.append([i, j])
                inds_prime.append([i, j + 1])
                labels.append(1)
                # make symmetric
                inds.append([i, j + 1])
                inds_prime.append([i, j])
                labels.append(1)

        shuffled_inds = np.random.rand(len(inds)).argsort()

        self.indices = np.concatenate([inds, inds])
        self.indices_prime = np.concatenate([inds_prime, np.array(inds_prime)[shuffled_inds]])
        self.labels = np.concatenate([labels, 2 * np.ones_like(labels)]).astype(np.float32)
        assert len(self.indices) == len(self.indices_prime)
        assert len(self.indices) == len(self.labels)

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


class Args(ParamsProto):
    env_id = "streetlearn"

    data_path = Proto("~/fair/streetlearn/processed-data/manhattan-medium",
                      help="path to the processed streetlearn dataset")
    street_view_size = Proto((64, 64), help="path to the streetlearn dataset", dtype=tuple)
    street_view_mode = Proto("omni-gray", help="OneOf[`omni-gray`, `ombi-rgb`]")

    seed = 0
    load_weights = Proto(None, help="path for trained weights", dtype=str)
    save_weights = Proto("models/local_metric.pkl", help="location to save the weight")

    latent_dim = 32
    local_metric = "LocalMetricConvDeep"

    num_epochs = 100
    batch_size = 16
    # lr = LinearAnneal(1e-3, min=1e-3, n=num_epochs + 1)
    lr = 1e-4
    beta_1 = 0.9
    beta_2 = 0.999
    weight_decay = Proto(0.01, help="should be close to zero.")

    # sample bias
    # K = Proto(1, help="The regularizing constant for how far we consider neighbors.")
    # subsample = Proto(1, help="sub-sample the trajectory")
    rest_subsample = Proto(1, help="sub-sample the trajectory")

    vis_k = Proto(10, help="The number of sample grid we show per epoch")
    vis_interval = 10
    vis_horizon = Proto(500, help="the horizon for visualizing score vs timesteps")

    checkpoint_interval = Proto(1, help="only training for 40 epochs, checkpoint every epoch.")
    checkpoint_after = Proto(-1, help="do not log from the start")


class DEBUG(ParamsProto):
    show_sample_trajectories = Proto(False, help="debug flag for saving sample trajectories")


def train(metric_fn):
    """Build dataset and train local metric for passive setting."""
    from os.path import expanduser

    # collect sample here
    from streetlearn import StreetLearnDataset
    streetlearn = StreetLearnDataset(expanduser(Args.data_path), Args.street_view_size, Args.street_view_mode)
    streetlearn.select_all()
    raw_images = [streetlearn.images[traj][..., None] for traj in streetlearn.trajs]

    dataset = PairDataset(raw_images)
    loader = DataLoader(dataset, batch_size=Args.batch_size, shuffle=True, num_workers=4)
    eval_loader = DataLoader(dataset, batch_size=Args.batch_size, shuffle=True, num_workers=4)
    score_distribution(dataset.labels, f"dataset/train_scores.png")

    from ml_logger import logger
    logger.log_text(f"""
        Distribution of Scores:

        0: {(dataset.labels == 0).sum():.0f}
        1: {(dataset.labels == 1).sum():.0f}
        2: {(dataset.labels == 2).sum():.0f}""", "dataset/label_balance.txt", dedent=True)

    # used for the score distribution stats. Make smaller for faster training
    # Note: trajectory contains coverage > 1. Images are repeated.
    all_images = torch.tensor(np.concatenate(raw_images).transpose(0, 3, 1, 2)[::1] / 255, dtype=torch.float32,
                              device=Args.device, requires_grad=False)
    traj_labels = torch.tensor(np.concatenate([np.ones(len(traj)) * i for i, traj in enumerate(raw_images)])[::1],
                               device=Args.device, requires_grad=False)

    optimizer = optim.Adam(metric_fn.parameters(), lr=Args.lr, betas=[Args.beta_1, Args.beta_2],
                           weight_decay=Args.weight_decay)

    # from torch_utils import RMSLoss
    # rms = RMSLoss()
    def evaluate(epoch, step):
        nonlocal all_images, traj_labels
        for i, ((s, s_prime), y) in enumerate(tqdm(eval_loader, total=len(eval_loader.dataset) // Args.batch_size)):
            if epoch == 0 and step == 0 and i == 0:
                logger.log_text(f"""
                      s.shape: {s.shape}
                s_prime.shape: {s_prime.shape}
                """, "debug/data_details.txt", dedent=True)
                logger.log_text(f"labels: {y}")
                logger.log_images(s.cpu().numpy().transpose(0, 2, 3, 1), n_cols=4, n_rows=4, key="debug/img.png")
                logger.log_images(s_prime.cpu().numpy().transpose(0, 2, 3, 1), n_cols=4, n_rows=4,
                                  key="debug/img_prime.png")

            with torch.no_grad():
                y_hat = metric_fn(s.to(Args.device), s_prime.to(Args.device)).squeeze()
            correct = lambda th: (y_hat.cpu() < th).numpy() == (y.cpu().numpy() < th)
            logger.store_metrics(metrics={f"eval/accuracy-{th}": correct(th) for th in [0.5, 1.1, 1.15, 1.25, 1.5]},
                                 y=y.mean(), sample_bias=y.numpy())

    num_batches = len(loader.dataset) // Args.batch_size

    start_checkpoint = Args.checkpoint_after or -1
    if start_checkpoint < 0:
        start_checkpoint = start_checkpoint + Args.num_epochs

    logger.split()
    for epoch in range(Args.num_epochs + 1):

        for i, ((s, s_prime), y) in enumerate(tqdm(loader, total=num_batches)):
            y_hat = metric_fn(s.to(Args.device), s_prime.to(Args.device)).squeeze()
            loss = F.smooth_l1_loss(y_hat, y.to(Args.device))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct = lambda th: (y_hat.cpu() < th).numpy() == (y.cpu().numpy() < th)
            with torch.no_grad():
                logger.store_metrics(metrics={f"accuracy-{th}": correct(th) for th in [0.5, 1.1, 1.15, 1.25, 1.5]},
                                     loss=loss.cpu().item(), y_hat=y_hat.mean().cpu().item(), y=y.mean().cpu().item(), )

            if i % 5000 == 0:
                evaluate(epoch, step=i // 5000)

            if i % 100 == 0:
                logger.log_metrics_summary(default_stat="quantile",
                                           key_values=dict(epoch=epoch + i / num_batches, dt_epoch=logger.split()))

        if epoch >= start_checkpoint and epoch % Args.checkpoint_interval == 0:
            with logger.SyncContext():  # use synchronized requests to make it fast.
                logger.save_module(
                    metric_fn, f"models/local_metric_{epoch:03d}.pkl", show_progress=True, chunk=1_000_000)


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

    f_local_metric = globals()[Args.local_metric](1, Args.latent_dim)
    f_local_metric.to(Args.device)
    logger.log_text(str(f_local_metric), "models/f_local_metric.txt")

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

    if False:
        thunk(main, seed=5 * 100)()
    else:

        cprint('Training on cluster', 'green')
        import jaynes

        jaynes.config("priority-gpu")
        for Args.lr in [1e-4, 3e-4, 1e-3]:
            _ = instr(main, seed=5 * 100, lr=Args.lr, epoch=200)
            jaynes.run(_)

        jaynes.listen()
