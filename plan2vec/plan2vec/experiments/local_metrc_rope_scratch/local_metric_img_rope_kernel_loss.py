import torch
from torch import optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from params_proto import cli_parse, Proto

from plan2vec.data_utils.rope_dataset import PairDataset, SeqDataset
from plan2vec.models.convnets import LocalMetricConvLarge
from plan2vec.plotting.rope_viz import score_distribution, top_neighbors, faraway_samples


def normalize(x):
    """Normalizes the

    :return: torchTensor(dtype=float64)
    """
    return (x - x.min()) / (x.max() - x.min())


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
    lr = 1e-3
    k_fold = Proto(5, dtype=int, help="The k-fold for validation")

    vis_k = Proto(10, help="The number of sample grid we show per epoch")
    vis_interval = 10
    vis_horizon = Proto(500, help="the horizon for visualizing score vs timesteps")


def train(metric_fn):
    """Build dataset and train local metric for passive setting."""
    from ml_logger import logger
    from os.path import expanduser

    rope = np.load(expanduser(Args.data_path))
    dataset = PairDataset(rope[len(rope) // Args.k_fold:])
    loader = DataLoader(dataset, batch_size=Args.batch_size, shuffle=True, num_workers=4)
    eval_dataset = PairDataset(rope[:len(rope) // Args.k_fold])
    eval_loader = DataLoader(eval_dataset, batch_size=Args.batch_size, shuffle=True, num_workers=4)

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
    epoch = 0

    def evaluate():
        nonlocal seq_gen, all_images, traj_labels
        for i, ((s, s_prime), is_neighbor) in enumerate(eval_loader):
            # generate the activation map
            y = 1 - is_neighbor
            if i < Args.vis_k and (epoch % Args.vis_interval == 0 or epoch < Args.vis_interval):
                s.requires_grad_(True)
                s_prime.requires_grad_(True)

                y_hat = metric_fn(s.to(Args.device), s_prime.to(Args.device)).squeeze()
                loss = F.smooth_l1_loss(y_hat, y.view(-1).to(Args.device))
                loss.backward()

                diff = normalize(s_prime - s)
                diff[:, :, :10, :10] = y[:, None, None, None]
                diff[:, :, :10, 10:20] = y_hat[:, None, None, None]
                _ = torch.cat([s, s_prime, diff,  # diff image
                               normalize(s.grad),  # activation of first image
                               normalize(s_prime.grad)], dim=1)
                stack = _.reshape(-1, *_.shape[2:])[:50].detach().cpu().numpy()
                logger.log_images(stack, f"figures/eval_pairs/epoch_{epoch:04d}/activation_{i:04d}.png", 5, 10)

            # visualize activation and samples
            with torch.no_grad():
                y_hat = metric_fn(s.to(Args.device), s_prime.to(Args.device)).squeeze()
                _ = torch.cat([s, s_prime, normalize(s_prime - s)], dim=1)
                stack = _.reshape(-1, *_.shape[2:])[:30].cpu().numpy()

                if i < Args.vis_k and epoch == 0:
                    logger.log_images(stack, f"figures/sample_pairs/{i:02d}.png", 5, 6)

            correct = (y_hat.cpu() > 0.5).numpy() == y.byte().cpu().numpy()
            pos_mask = y.byte().cpu().numpy() == 1
            logger.store_metrics(
                metrics={"accuracy": correct,
                         "accuracy/0": correct[1 - pos_mask],
                         "accuracy/1": correct[pos_mask], },
                y=y.mean(),
                sample_bias=y.numpy())

        # score vs timesteps
        with torch.no_grad():
            y_hat = metric_fn(eval_x, eval_trajs).squeeze()
            from plan2vec.plotting.visualize_traj_2d import local_metric_over_trajectory
            local_metric_over_trajectory(y_hat.cpu(), f"figures/epoch_{epoch:04d}/score_vs_timesteps_{epoch:04d}.png")
            score_distribution(y_hat.cpu(), f"figures/epoch_{epoch:04d}/in_trajectory_scores.png", xlim=[-.1, .1],
                               ylim=[0, 100])
            score_distribution(y_hat.cpu(), f"figures/epoch_{epoch:04d}/in_trajectory_scores_full.png", xlim=[-.1, 1.1],
                               ylim=[0, 100])

            y_hat = metric_fn(all_images[:, None, :, :, :], all_images[None, :1, :, :, :]).squeeze()
            score_distribution(y_hat.cpu(), f"figures/epoch_{epoch:04d}/all_scores.png", xlim=[-.1, .1], ylim=[0, 100])
            score_distribution(y_hat.cpu(), f"figures/epoch_{epoch:04d}/all_scores_full.png", xlim=[-.1, 1.1],
                               ylim=[0, 100])
            top_neighbors(all_images, all_images[0, :, :, :], y_hat, f"figures/epoch_{epoch:04d}/top_neighbors.png")
            faraway_samples(all_images, all_images[0, :, :, :], y_hat, f"figures/epoch_{epoch:04d}/faraway_samples.png")

    logger.split()
    for epoch in range(Args.num_epochs + 1):
        evaluate()

        for i, ((s, s_prime), is_neighbor) in enumerate(loader):
            y = 1 - is_neighbor
            y_hat = metric_fn(s.to(Args.device), s_prime.to(Args.device))
            loss = F.smooth_l1_loss(y_hat.view(-1), y.view(-1).to(Args.device))
            logger.store_metrics(loss=loss.cpu().item())

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.log_metrics_summary(key_values=dict(epoch=epoch, dt_epoch=logger.split()),
                                   default_stats='mean')

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

    if True:
        instr(main, seed=5 * 100, __postfix="local-metric-experiments")

    elif False:
        cprint('Training on cluster', 'green')
        import jaynes

        # jaynes.config("devfair")
        jaynes.config("learnfair-gpu")
        # jaynes.config("dev-gpu")

        _ = thunk(main, seed=5 * 100, __prefix="local-metric-experiments")
        jaynes.run(_)
        jaynes.listen()
