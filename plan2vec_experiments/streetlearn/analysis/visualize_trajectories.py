from params_proto import cli_parse, Proto
from tqdm import tqdm


@cli_parse
class Args:
    data_path = Proto("~/fair/streetlearn/processed-data/manhattan-medium",
                      help="path to the processed streetlearn dataset")
    street_view_size = Proto((64, 64), help="image size for the dataset", dtype=tuple)
    street_view_mode = Proto("omni-gray", help="OneOf[`omni-gray`, `ombi-rgb`]")

    exp_path = "/episodeyang/plan2vec/2019/05-20/streetlearn/manhattan-medium/1-step/23.46/57.577772"

    term_r = 1.5
    load_local_metric = None

    device = None


def connect_the_dots(all_images, lng_lats, f_local_metric):
    """
    The key here, is that we want to show connections that come from different trajectories.

    :param all_images:
    :param trajs:
    :param f_local_metric:
    :return:
    """
    import numpy as np
    import torch
    from ml_logger import logger

    with torch.no_grad():
        from plan2vec.plan2vec.plan2vec_streetlearn import pairwise_fn

        magic = [1, 0.75]  # projective correction

        _ = torch.tensor(all_images, device=Args.device, dtype=torch.float32)
        logger.print('compute pairwise distance...', color="green")
        pairwise_ds = pairwise_fn(f_local_metric, _, _, chunk=1)
        logger.print("shape of the pairwise matrix: ", pairwise_ds.shape)

        pairwise_ds[torch.eye(len(all_images), dtype=torch.uint8)] = float('inf')
        _ = (lng_lats[None, :, :] - lng_lats[:, None, :]) / magic
        gt_ds = np.linalg.norm(_, ord=2, axis=-1)

        ns_mask = pairwise_ds < Args.term_r
        mask = np.ma.make_mask(ns_mask.cpu().numpy())

        _ = np.arange(len(all_images))
        pair_indices = np.stack(np.meshgrid(_, _)).transpose(2, 1, 0)[mask]

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(3.5, 2.5), dpi=140)
    # put your plotting code here.
    plt.scatter(gt_ds, pairwise_ds.cpu().numpy(),
                alpha=0.5, facecolor="#23aaff", edgecolor="none", label="all")
    plt.scatter(gt_ds[mask],
                pairwise_ds[ns_mask].cpu().numpy(),
                alpha=0.5, facecolor="red", edgecolor="none", label=f"φ < {Args.term_r:.02f}".rstrip('0').rstrip('.'))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylim(-0.1, 2.1)
    plt.xlim(-0.01, 0.03)
    plt.legend(loc="upper left", bbox_to_anchor=(0.45, 0.8), framealpha=1, frameon=False, fontsize=12)
    logger.savefig(f'figures/streetlearn_local_metric_score_vs_gt.png', fig=fig, overwrite=True)
    plt.close()

    fig = plt.figure(figsize=(3, 3), dpi=140)

    # put your plotting code here.
    lines = lng_lats[pair_indices].transpose(0, 2, 1)
    for ind, l in enumerate(lines):
        plt.plot(*l, c="red", lw=0.5, alpha=0.2 + 0.8 * ind / len(lines))

    # # magic box and sizes
    # bbox = (-73.997, 40.726, 0.01, 0.008)
    # plt.xlim(bbox[0] - 0.0015, bbox[0] + bbox[2] + 0.0015)
    # plt.ylim(bbox[1] - 0.001, bbox[1] + bbox[3] + 0.001)
    logger.savefig(f'figures/streetlearn_connect_the_dots.png', fig=fig)
    plt.close()


def main():
    import torch
    import numpy as np
    from ml_logger import logger
    from os.path import expanduser
    import matplotlib.pyplot as plt
    from more_itertools import chunked
    from plan2vec.models.convnets import LocalMetricConvDeep

    Args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log_params(Args=vars(Args))

    # loading the metric function
    f_local_metric = LocalMetricConvDeep(None, None)
    f_local_metric.to(Args.device)
    f_local_metric.eval()
    logger.load_module(f_local_metric, Args.load_local_metric)

    # loading the dataset
    from streetlearn import StreetLearnDataset
    streetlearn = StreetLearnDataset(expanduser(Args.data_path), Args.street_view_size, Args.street_view_mode)
    streetlearn.select_all()
    # raw_images = [streetlearn.images[traj][..., None] for traj in streetlearn.trajs]

    all_images = streetlearn.images[:, None, ...] / 255

    connect_the_dots(all_images[::20], streetlearn.lng_lat[::20], f_local_metric)


def vis_trajectories():
    import numpy as np
    from ml_logger import logger
    import matplotlib.pyplot as plt

    with logger.PrefixContext(Args.exp_path):
        traj_paths = sorted(logger.glob('**/traj*.pkl', wd=logger.prefix))
        trajs = [logger.load_pkl(p, tries=5) for p in tqdm(traj_paths, desc="loading trajs...")]

    with logger.PrefixContext(Args.exp_path):
        traj, = logger.load_pkl(traj_paths[-10], tries=5)

    s = traj['s']  # .transpose(1, 0, 2)
    s_goal = traj['s_goal']  # .transpose(1, 0, 2)
    inds = np.arange(len(s))
    colors = np.zeros((len(s), 4))
    colors[:, 3] = inds / len(s)

    plt.scatter(*s[:, 0].T, c=colors, s=1, edgecolors='none', lw=1)
    # plt.scatter(*s_goal[:, 0].T, color='red')

    _ = s[:, 0] - s[0, 0]
    delta = s[:-1, 0] - s[1:, 0]
    delta_ds = np.linalg.norm(delta, ord=2, axis=-1)
    plt.hist(delta_ds)
    plt.show()

    plt.scatter(*_.T)
    plt.show()

    bbox = (-73.997, 40.726, 0.02, 0.016)
    plt.xlim(bbox[0] - 0.003, bbox[0] + bbox[2] + 0.003)
    plt.ylim(bbox[1] - 0.001, bbox[1] + bbox[3] + 0.001)
    plt.show()


def visualize_embedding():
    import torch
    import numpy as np
    from ml_logger import logger
    from os.path import expanduser
    import matplotlib.pyplot as plt
    from more_itertools import chunked
    from plan2vec.models.convnets import GlobalMetricConvDeepL2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading the metric function
    global_metric = GlobalMetricConvDeepL2(1, 2)
    global_metric.to(device)
    global_metric.eval()

    with logger.PrefixContext(Args.exp_path):
        model_paths = sorted(logger.glob('**/global_metric*.pkl'))
        last = model_paths[-1]

        logger.load_module(global_metric, last)

    # loading the dataset
    from streetlearn import StreetLearnDataset
    streetlearn = StreetLearnDataset(expanduser(Args.data_path), Args.street_view_size, Args.street_view_mode)
    streetlearn.select_all()
    raw_images = [streetlearn.images[traj][..., None] for traj in streetlearn.trajs]

    all_images = np.concatenate(raw_images).transpose(0, 3, 1, 2) / 255

    with torch.no_grad():
        labels = torch.cat([global_metric.encode(torch.tensor(chunk, dtype=torch.float32, device=device))
                            for chunk in tqdm(chunked(all_images, 10000), desc='evaluating streetviews')])

    plt.scatter(*labels.cpu().numpy().T)
    plt.show()


if __name__ == "__main__":
    import jaynes
    from plan2vec_experiments import instr

    # exp_path = "episodeyang/plan2vec/2019/05-20/streetlearn/local_metric/18.12/08.412698"
    exp_path = "episodeyang/plan2vec/2019/05-21/streetlearn/local_metric/11.16/58.274405"

    Args.load_local_metric = f"/{exp_path}/models/local_metric_100.pkl"

    jaynes.config('vector-gpu')

    jaynes.run(instr(main))
    jaynes.listen()

    # _ = thunk(vis_trajectories)()
    # _ = thunk(main)()
