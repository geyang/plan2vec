import torch
import numpy as np
from params_proto import cli_parse

local_metric_exp_path = "episodeyang/plan2vec/2019/05-07/c-maze-image/c_maze_local_metric/21.19/10.332015"


@cli_parse
class Args:
    seed = 200
    env_id = 'CMazeDiscreteImgIdLess-v0'
    n_rollouts = 120
    steps = 4
    load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


GOALS = {
    'CMazeDiscreteImgIdLess-v0': np.array([-0.15, -0.15]),
    'GoalMassDiscreteImgIdLess-v0': np.array([0.15, -0.15]),
}


def make_trajs():
    from tqdm import trange
    from ge_world import gym

    env = gym.make(Args.env_id)

    env.seed(Args.seed)
    np.random.seed(Args.seed)

    trajs = []
    img_trajs = []
    for _ in trange(Args.n_rollouts, desc="rolling out"):
        obs = env.reset(goal=GOALS[Args.env_id])
        traj = [obs['x']]
        img_traj = [obs['img']]
        for step in range(Args.steps):
            act = np.random.randint(8)
            obs, reward, done, _ = env.step(act)
            traj.append(obs['x'])
            img_traj.append(obs['img'])

        trajs.append(traj)
        img_trajs.append(img_traj)

    env.close()

    return np.array(trajs), np.array(img_trajs)


def connect_the_dots():
    from plan2vec.models.convnets import LocalMetricConvLarge
    from ml_logger import logger
    from torch_utils import torchify
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    if not logger.prefix:
        from plan2vec_experiments import RUN
        logger.configure(RUN.server, "analysis/c_maze_dataset", register_experiment=False)

    Args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f_local_metric = LocalMetricConvLarge(1, Args.latent_dim).to(Args.device)
    f_local_metric.eval()
    logger.load_module(f_local_metric, Args.load_local_metric)

    trajs, img_trajs = make_trajs()

    all_trajs = trajs.reshape(-1, 2)
    all_img_trajs = img_trajs.reshape(-1, 1, 64, 64)
    N, H, *_ = trajs.shape
    traj_labels = np.tile(np.arange(N)[:, None], [1, H]).reshape(-1)
    in_traj_inds = np.tile(np.arange(H)[None, :], [N, 1]).reshape(-1)

    with torch.no_grad():
        from plan2vec.plan2vec.plan2vec_img import pairwise_fn
        _ = torch.tensor(all_img_trajs, device=Args.device, dtype=torch.float32)
        pairwise_ds = pairwise_fn(f_local_metric, _, _, chunk=3)
        print(pairwise_ds.shape)
        gt_ds = np.linalg.norm(all_trajs[:, None, :] - all_trajs[None, :, :], ord=2, axis=-1)
        pairwise_ds[torch.eye(N * H, dtype=torch.uint8)] = float('inf')
        ns_mask = (pairwise_ds < 1).cpu().numpy()
        mask = np.ma.make_mask(ns_mask)

    conns = []
    for traj_ind, x, row_mask in zip(traj_labels, all_trajs, ns_mask):
        m = np.ma.make_mask(row_mask)
        ns = all_trajs[m]
        ns_traj_ind = traj_labels[m]
        for n, _traj_ind in zip(ns, ns_traj_ind):
            if traj_ind != _traj_ind:
                conns.append([x, n])
    conns = np.array(conns)

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
    logger.savefig("figures/c_maze_img.png", bbox_inches=0, pad_inches=0)
    plt.close()
    # fig.show()

    # plot here
    rcParams['axes.titlepad'] = 15
    rcParams['axes.labelpad'] = 12
    rcParams['axes.labelsize'] = 14

    plt.figure(figsize=(4, 3), dpi=300)
    plt.title("C-Maze Local Metric")
    plt.scatter(gt_ds[::200], pairwise_ds.cpu().numpy()[::200],
                alpha=0.5, facecolor="#23aaff", edgecolor="none", label="all")
    plt.scatter(gt_ds[mask][::50], pairwise_ds[ns_mask].cpu().numpy()[::50],
                alpha=0.5, facecolor="red", edgecolor="none", label="φ < 1")
    plt.xlim(-0.02, 0.2)
    plt.ylim(-0.1, 2.4)
    plt.ylabel("Score φ")
    plt.xlabel("Ground-truth L2 Distance")
    plt.legend(loc="upper left", bbox_to_anchor=(0.4, 0.5), framealpha=1, frameon=False, fontsize=13,
               handletextpad=-0.4)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    logger.savefig("figures/c_maze_img_score_vs_gt.png", bbox_inches='tight')
    plt.close()
    # fig.show()


if __name__ == "__main__":
    from plan2vec_experiments import instr

    import jaynes

    jaynes.config('devfair')
    jaynes.run(instr(connect_the_dots))
    jaynes.listen()

    # for Args.env_id in ["CMazeDiscreteIdLess-v0", "GoalMassDiscreteIdLess-v0", ]:
    #     plot_trajs()
