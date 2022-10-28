import numpy as np
import torch
from params_proto import cli_parse


@cli_parse
class Args:
    seed = 200
    latent_dim = 50

    order = 2

    term_r = 1
    # env_id = '回MazeDiscreteIdLess-v0'
    env_id = 'GoalMazeDiscreteIdLess-v0'
    n_rollouts = 120
    steps = 30


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

    return np.array(trajs)  # , np.array(steps)


def ground_truth_vs_timesteps(trajs):
    # from plan2vec.mdp.models import LocalMetric
    from ml_logger import logger
    from torch_utils import torchify
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    all_trajs = trajs.reshape(-1, 2)
    N, H, *_ = trajs.shape
    traj_labels = np.tile(np.arange(N)[:, None], [1, H]).reshape(-1)
    step_labels = np.tile(np.arange(H)[None, :], [N, 1]).reshape(-1)

    # for within each trajectory of length K, we can compute k * (k - 1) pairs, with
    # variying timesteps between then. Here is how we construct these pairs:
    pairs, delta_ts = [], []
    for traj in trajs:
        l = len(traj)
        for k in range(l):
            for i in range(0, l - k):
                pairs.append([traj[i], traj[i + k]])
                delta_ts.append(np.abs(step_labels[i] - step_labels[i + k]))

    delta_ts = np.array(delta_ts)
    pairs = np.array(pairs)
    print(delta_ts.shape, pairs.shape)

    with torch.no_grad():
        gt_ds = np.linalg.norm(pairs[:, 0] - pairs[:, 1], ord=Args.order, axis=-1)

    print(gt_ds.shape)

    # plot here
    rcParams['axes.titlepad'] = 15
    rcParams['axes.labelpad'] = 12
    rcParams['axes.labelsize'] = 14

    fig = plt.figure(figsize=(4, 3), dpi=300)
    plt.title(title[Args.env_id].title() + " Trajectories")
    plt.scatter(delta_ts, gt_ds,  # pairwise_ds.numpy()[::200],
                alpha=0.01, facecolor="#23aaff", edgecolor="none", label="all")
    plt.ylim(-0.05, 0.4)
    plt.ylabel(f"L{Args.order:.0f} Distance")
    plt.xlabel("Timesteps in-between")

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    logger.savefig(f"figures/{title[Args.env_id]}_distance_vs_timesteps_wide.png", bbox_inches='tight', fig=fig)

    plt.xlim(-0.02, 5)
    logger.savefig(f"figures/{title[Args.env_id]}_distance_vs_timesteps.png", bbox_inches='tight', fig=fig)


def main(**kwargs):
    from ml_logger import logger
    if not logger.prefix:
        from plan2vec_experiments import RUN
        logger.configure(RUN.server, f"analysis/{logger.stem(__file__)}_dataset", register_experiment=False)

    Args.update(kwargs)
    logger.log_params(Args=vars(Args))

    trajs = make_trajs()

    ground_truth_vs_timesteps(trajs)


if __name__ == "__main__":
    Args.env_id = "GoalMassDiscreteIdLess-v0"
    main(**vars(Args))
