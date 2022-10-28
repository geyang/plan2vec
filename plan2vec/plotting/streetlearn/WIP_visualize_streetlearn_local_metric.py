from os.path import exists

import numpy as np
import torch
from params_proto import cli_parse, Proto
from tqdm import tqdm


@cli_parse
class Args:
    seed = 200
    data_path = Proto("~/fair/streetlearn/processed-data/manhattan-tiny",
                      help="path to the processed streetlearn dataset")
    street_view_size = Proto((64, 64), help="image size for the dataset", dtype=tuple)
    street_view_mode = Proto("omni-gray", help="OneOf[`omni-gray`, `ombi-rgb`]")

    term_r = 1
    n_rollouts = 120
    steps = 4
    load_local_metric = None


GOALS = {
    'CMazeDiscreteImgIdLess-v0': np.array([-0.15, -0.15]),
    'GoalMassDiscreteImgIdLess-v0': np.array([0.15, -0.15]),
    '回MazeDiscreteImgIdLess-v0': np.array([0.15, -0.05]),
}

title = {
    'CMazeDiscreteImgIdLess-v0': "c-maze",
    'GoalMassDiscreteImgIdLess-v0': "goal-mass",
    '回MazeDiscreteImgIdLess-v0': "回-maze"
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

    env.reset(x=np.array([-0.16, 0.15]), goal=GOALS[Args.env_id])
    image = env.render('rgb', width=640, height=640)
    env.close()

    from ml_logger import logger
    logger.log_image(image, f"figures/{title[Args.env_id]}_render.png")

    return np.array(trajs), np.array(img_trajs)


def connect_the_dots(trajs, img_trajs, f_local_metric=None):
    from plan2vec.models.convnets import LocalMetricConvLarge
    from ml_logger import logger
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.font_manager import FontProperties

    Args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if f_local_metric is None:
        f_local_metric = LocalMetricConvLarge(1, Args.latent_dim).to(Args.device)
        logger.load_module(f_local_metric, Args.load_local_metric)
        f_local_metric.eval()

    all_trajs = trajs.reshape(-1, 2)
    all_img_trajs = img_trajs.reshape(-1, 1, 64, 64)
    N, H, *_ = trajs.shape

    traj_labels = np.tile(np.arange(N)[:, None], [1, H]).reshape(-1)

    with torch.no_grad():
        from plan2vec.plan2vec.plan2vec_img import pairwise_fn
        _ = torch.tensor(all_img_trajs, device=Args.device, dtype=torch.float32)
        logger.print('compute pairwise distance...', color="green")
        pairwise_ds = pairwise_fn(f_local_metric, _, _, chunk=3)
        logger.print("shape of the pairwise matrix: ", pairwise_ds.shape)
        gt_ds = np.linalg.norm(all_trajs[:, None, :] - all_trajs[None, :, :], ord=2, axis=-1)
        pairwise_ds[torch.eye(N * H, dtype=torch.uint8)] = float('inf')
        ns_mask = pairwise_ds < Args.term_r
        mask = np.ma.make_mask(ns_mask.cpu().numpy())

    data_size = gt_ds.shape[0]

    # plot here
    rcParams['axes.titlepad'] = 15
    rcParams['axes.labelpad'] = 12
    rcParams['axes.labelsize'] = 14

    fig = plt.figure(figsize=(4, 3), dpi=300)

    CJK_font_path = '/private/home/geyang/opentype/noto/NotoSansCJK-Regular.ttc'
    if exists(CJK_font_path):
        plt.title(f"{title[Args.env_id].title()} Local Metric", fontproperties=FontProperties(fname=CJK_font_path))
    else:
        plt.title(f"{title[Args.env_id].title()} Local Metric", )

    plt.scatter(gt_ds[::200 if data_size > 500 else 4],
                pairwise_ds.cpu().numpy()[::200 if data_size > 500 else 4],
                alpha=0.5, facecolor="#23aaff", edgecolor="none", label="all")
    plt.scatter(gt_ds[mask][::50 if data_size > 500 else 1],
                pairwise_ds[ns_mask].cpu().numpy()[::50 if data_size > 500 else 1],
                alpha=0.5, facecolor="red", edgecolor="none", label=f"φ < {Args.term_r:.02f}".rstrip('0').rstrip('.'))
    plt.ylabel("Score φ")
    plt.xlabel("Ground-truth L2 Distance")
    plt.legend(loc="upper left", bbox_to_anchor=(0.6, 0.5), framealpha=1, frameon=False, fontsize=13,
               handletextpad=-0.4)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylim(-0.1, 2.4)

    plt.xlim(-0.02, 0.2)
    logger.savefig(f"figures/{title[Args.env_id]}_score_vs_gt.png", bbox_inches='tight', fig=fig)

    plt.xlim(-0.02, 0.6)
    logger.savefig(f"figures/{title[Args.env_id]}_score_vs_gt_wider.png", bbox_inches='tight', fig=fig)
    plt.close()
    logger.print('saving score_vs_gt figure', color="green")

    conns = []
    for traj_ind, x, row_mask in tqdm(zip(traj_labels, all_trajs, mask), desc="find edges..."):
        ns = all_trajs[row_mask]
        ns_traj_ind = traj_labels[row_mask]
        for n, _traj_ind in zip(ns, ns_traj_ind):
            if traj_ind != _traj_ind:
                conns.append([x, n])
    conns = np.array(conns)
    logger.print("number of connections: ", conns.size)

    fig = plt.figure(figsize=(3, 3), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim([-0.3, 0.3])
    ax.set_ylim([-0.3, 0.3])
    ax.set_aspect('equal')

    for i, traj in enumerate(tqdm(trajs)):
        ax.plot(traj[:, 0], traj[:, 1], c="#23aaff", linewidth=4, alpha=i * 0.7 / len(trajs))
        ax.scatter(trajs[i, 0, 0], trajs[i, 0, 1], s=0.5, color='gray', zorder=1, alpha=1)

    logger.print('saving dataset figure', color="green")
    logger.savefig(f"figures/{title[Args.env_id]}_data.png", bbox_inches=0, pad_inches=0, fig=fig)

    for i, conn in enumerate(tqdm(conns)):
        ax.plot(conn[:, 0], conn[:, 1], color='red', linewidth=1, alpha=0.3 * i / len(conns))

    logger.print('saving connected figure', color="green")
    logger.savefig(f"figures/{title[Args.env_id]}_connected.png", bbox_inches=0, pad_inches=0, fig=fig)

    plt.close()


def main(**kwargs):
    from ml_logger import logger
    if not logger.prefix:
        from plan2vec_experiments import RUN
        logger.configure(RUN.server, f"analysis/{logger.stem(__file__)}_dataset", register_experiment=False)

    Args.update(kwargs)
    logger.log_params(Args=vars(Args))

    trajs, img_trajs = make_trajs()

    connect_the_dots(trajs, img_trajs)


if __name__ == "__main__":
    from plan2vec_experiments import instr
    import jaynes

    # jaynes.config('learnfair-gpu')
    jaynes.config('local')

    exp_path = ""
    jaynes.run(instr(main, load_local_metric=f"/{exp_path}/models/local_metric.pkl"))

    jaynes.listen()
