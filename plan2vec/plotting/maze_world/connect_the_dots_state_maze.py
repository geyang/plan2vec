import numpy as np
import torch
from params_proto import cli_parse


@cli_parse
class Args:
    seed = 200
    latent_dim = 50

    term_r = 1
    env_id = '回MazeDiscreteIdLess-v0'
    n_rollouts = 120
    steps = 4
    load_local_metric = None


GOALS = {
    'CMazeDiscreteIdLess-v0': np.array([-0.15, -0.15]),
    'GoalMassDiscreteIdLess-v0': np.array([0.15, -0.15]),
    '回MazeDiscreteIdLess-v0': np.array([-0.15, 0.05]),
}

title = {
    'CMazeDiscreteIdLess-v0': "c-maze",
    'GoalMassDiscreteIdLess-v0': "goal-mass",
    '回MazeDiscreteIdLess-v0': "回-maze"
}


def make_trajs():
    from tqdm import trange
    from ge_world import gym

    env = gym.make(Args.env_id)

    env.seed(Args.seed)
    np.random.seed(Args.seed)

    trajs = []
    for _ in trange(Args.n_rollouts, desc="rolling out"):
        obs = env.reset(goal=GOALS[Args.env_id])
        traj = [obs['x']]
        for step in range(Args.steps):
            act = np.random.randint(8)
            obs, reward, done, _ = env.step(act)
            traj.append(obs['x'])
        trajs.append(traj)

    env.close()

    return np.array(trajs)


def connect_the_dots(trajs):
    from plan2vec.models.mlp import LocalMetric
    from ml_logger import logger
    from torch_utils import torchify
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    f_local_metric = LocalMetric(2, Args.latent_dim).to(Args.device)
    f_local_metric.eval()
    logger.load_module(f_local_metric, Args.load_local_metric)

    φ = torchify(f_local_metric, input_only=True, dtype=torch.float32)

    all_trajs = trajs.reshape(-1, 2)
    N, H, *_ = trajs.shape
    traj_labels = np.tile(np.arange(N)[:, None], [1, H]).reshape(-1)
    in_traj_inds = np.tile(np.arange(H)[None, :], [N, 1]).reshape(-1)

    with torch.no_grad():
        pairwise_ds = φ(all_trajs[:, None, :], all_trajs[None, :, :]).squeeze()
        gt_ds = np.linalg.norm(all_trajs[:, None, :] - all_trajs[None, :, :], ord=2, axis=-1)
        pairwise_ds[torch.eye(N * H, dtype=torch.uint8)] = float('inf')
        ns_mask = pairwise_ds < Args.term_r
        mask = np.ma.make_mask(ns_mask)
        from copy import copy

    conns = []
    for traj_ind, x, row_mask in zip(traj_labels, all_trajs, ns_mask):
        m = np.ma.make_mask(row_mask)
        ns = all_trajs[m]
        ns_traj_ind = traj_labels[m]
        for n, _traj_ind in zip(ns, ns_traj_ind):
            if traj_ind != _traj_ind:
                conns.append([x, n])
    conns = np.array(conns)

    rcParams['axes.titlepad'] = 0
    rcParams['axes.labelpad'] = 0
    rcParams['axes.labelsize'] = 0

    fig = plt.figure(figsize=(3, 3), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    for i, traj in enumerate(trajs):
        ax.plot(traj[:, 0], traj[:, 1], c="#23aaff", linewidth=4, alpha=i * 0.7 / len(trajs))
        ax.scatter(trajs[i, 0, 0], trajs[i, 0, 1], s=0.5, color='gray', zorder=1, alpha=1)
    for i, conn in enumerate(conns):
        ax.plot(conn[:, 0], conn[:, 1], color='red', linewidth=1, alpha=0.3 * i / len(conns))

    plt.xlim([-0.3, 0.3])
    plt.ylim([-0.3, 0.3])
    ax.set_aspect('equal')
    logger.savefig(f"figures/{title[Args.env_id]}_state.png", bbox_inches=0, pad_inches=0)
    plt.close()
    # fig.show()

    # plot here
    rcParams['axes.titlepad'] = 15
    rcParams['axes.labelpad'] = 12
    rcParams['axes.labelsize'] = 14

    fig = plt.figure(figsize=(4, 3), dpi=300)
    plt.title("C-Maze Local Metric")
    plt.scatter(gt_ds[::200], pairwise_ds.numpy()[::200],
                alpha=0.5, facecolor="#23aaff", edgecolor="none", label="all")
    plt.scatter(gt_ds[mask][::50], pairwise_ds[ns_mask].numpy()[::50],
                alpha=0.5, facecolor="red", edgecolor="none", label=f"φ < {Args.term_r}")
    plt.ylim(-0.1, 2.4)
    plt.ylabel("Score φ")
    plt.xlabel("Ground-truth L2 Distance")
    plt.legend(loc="upper left", bbox_to_anchor=(0.4, 0.5), framealpha=1, frameon=False, fontsize=13,
               handletextpad=-0.4)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.xlim(-0.02, 0.2)
    logger.savefig(f"figures/{title[Args.env_id]}_score_vs_gt.png", bbox_inches='tight', fig=fig)

    plt.xlim(-0.02, 0.6)
    logger.savefig(f"figures/{title[Args.env_id]}_score_vs_gt_wider.png", bbox_inches='tight', fig=fig)
    plt.close()


def main(**kwargs):
    from ml_logger import logger
    if not logger.prefix:
        from plan2vec_experiments import RUN
        logger.configure(RUN.server, f"analysis/{logger.stem(__file__)}_dataset", register_experiment=False)

    Args.update(kwargs)
    logger.log_params(Args=vars(Args))

    trajs = make_trajs()

    connect_the_dots(trajs)
