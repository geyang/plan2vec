import math
import torch
import numpy as np
from tqdm import trange
from functools import lru_cache
from params_proto import proto_partial
from params_proto.neo_proto import ParamsProto, Proto

from plan2vec.models.convnets import LocalMetricConvDeep
from plan2vec.plan2vec.plan2vec_rope import Args, pairwise_fn


class Config(ParamsProto):
    # for computing the pairwise distance matrix
    # exp_path = "/geyang/plan2vec/2019/12-17/analysis/local-metric-analysis/all_local_metric/21.33/rope/K-(1)/47.355423"
    exp_path = "/geyang/plan2vec/2020/01-11/analysis/local-metric-analysis/all_local_metric/20.59/rope/K-(1)/31.165902"
    weight_file = "models/local_metric_000-005.pkl"

    n_chunks = 20

    # for making the plots
    # chunk_prefix = "/geyang/plan2vec/2019/12-17/rope/rope_pairwise/map_reduce/21.40/51.465418"
    chunk_prefix = "/geyang/plan2vec/2020/01-12/rope/rope_pairwise/map_reduce/17.29/48.015289"
    start = 200
    # threshold = 1.4
    threshold = 0.5


def pairwise(k, n):
    """worker function for computing the pairwise distance between samples.

    :param k: the index for this run
    :param n: the total number of slices
    :return:
    """
    from ml_logger import logger

    logger.log_line(f"<main>(k: {k}, n: {n})", color='yellow')

    Args.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    Args.load_local_metric = f"{Config.exp_path}/{Config.weight_file}"

    if True:  # load local metric
        logger.log_line('loading local metric', end="... ", color="yellow")

        # hard code args for experiment
        f_local_metric = LocalMetricConvDeep(1, 32).to(Args.device)
        logger.load_module(f_local_metric, Args.load_local_metric)
        logger.log_line('✔done', color='green')

    if True:  # get rope dataset
        from os.path import expanduser

        logger.log_line('loading rope dataset', color="yellow", end="... ")
        rope = np.load(expanduser(Args.data_path), allow_pickle=True)[:10]
        all_images = np.concatenate(rope).transpose(0, 3, 1, 2)
        traj_labels = np.concatenate([np.ones(len(traj)) * i for i, traj in enumerate(rope)])
        logger.log_line('✔done', color='green')

    chunk_len = math.ceil(len(all_images) / n)
    start, end = chunk_len * k, chunk_len * (k + 1)
    logger.log_line(f"chunk_{k:02d}/{n} => slice [{start}:{end}]")

    with torch.no_grad():
        xs = torch.tensor(all_images / 255, device=Args.device, dtype=torch.float32)
        ds_slice = pairwise_fn(φ=f_local_metric, xs=xs, xs_slice=xs[start:end], chunk=1)
        logger.log_data(ds_slice.cpu().numpy(), f"chunk_{k:02d}.pkl")

    logger.log_line(f"chunk_{k:02d}.pkl is now saved!", color="green")


def map_reduce():
    from ml_logger import logger
    from plan2vec_experiments import instr

    thunk = instr(pairwise, __prefix="map_reduce", __postfix=logger.now("%S.%f"), __no_timestamp=True)

    for k in trange(Config.n_chunks):
        jaynes.run(thunk, k=k, n=Config.n_chunks)
        logger.print(f'{k}/{Config.n_chunks} is now launched')

    jaynes.listen()


def topk(chunk_prefix=None, single_traj=None):
    from os.path import expanduser
    from tqdm import trange
    import matplotlib.pyplot as plt
    import numpy as np
    from ml_logger import logger
    from plan2vec_experiments import RUN

    # %%
    n = 20
    k = 24

    logger.configure(RUN.server, prefix=chunk_prefix, register_experiment=False)
    logger.log_line(f"----------- saving the top {k} neighbors -----------")
    ds = np.concatenate([logger.load_pkl(f"chunk_{k:02d}.pkl")[0] for k in trange(n, desc="load")])
    # logger.log_data(ds, f"pairwise.npy")
    # np.save(expanduser("~/fair/pairwise.npy"), ds)

    rope = np.load(expanduser("~/fair/new_rope_dataset/data/new_rope.npy"), allow_pickle=True)[:10]
    all_images = np.concatenate(rope).transpose(0, 3, 1, 2)
    traj_labels = np.concatenate([np.ones(len(traj)) * i for i, traj in enumerate(rope)]).astype(int)

    print(len(all_images), len(ds))
    assert len(all_images) == len(ds), "the dimension of the distance matrix and the images should agree."
    logger.log_line('rope dataset is now loaded', color="yellow")

    with torch.no_grad():
        _ds = torch.tensor(ds)
        _ds[torch.eye(ds.shape[0], dtype=torch.uint8)] = float('inf')
        if single_traj is not None:
            # done: pick only from a particular trajectory, by filtering out other trajs.
            s, e = np.argmax(traj_labels == single_traj), np.argmax(traj_labels == single_traj + 1)
            logger.log_line(f"trajectory {single_traj} => [s: {s} e: {e}]", color="green", file=f"top-{k}.md")
            _ds[:, :s] = float('inf')
            _ds[:, e:] = float('inf')
        top_cols, top_col_inds = torch.topk(_ds, k=k, dim=1, largest=False, sorted=True)

        with logger.SyncContext():
            logger.log_data(data=dict(top=top_cols.numpy(), inds=top_col_inds.numpy()),
                            path=f"top_{k}.pkl" if single_traj is None else f"top_{k}-{single_traj:02d}",
                            overwrite=True)

    # %%
    fig = plt.figure(figsize=(4, 3))
    plt.title("Distribution of scores for top-24 neighbors")
    plt.hist(top_cols.numpy().flatten(), bins=80, histtype="step", density=True, linewidth=4, color="red", alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # logger.savefig(fig=fig, key=f"figures/score_distribution_top-{k}.png")
    # plt.savefig(fig=fig, fname=f"figures/score_distribution_top-{k}.png")
    fig.show()
    plt.close()
    # %%
    fig = plt.figure(figsize=(6, 3))
    plt.title("Scores Distribution")
    plt.hist(top_cols.numpy().flatten(), bins=80, histtype="step", density=True, linewidth=4, color="red", alpha=0.5,
             label=f"top-{k}")
    plt.hist(ds.flatten(), bins=80, histtype="step", density=True, linewidth=4, color="#23aaff", alpha=0.5,
             range=[0, 2], label="all")
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(loc="upper left", bbox_to_anchor=(0.95, 0.7), framealpha=1, frameon=False, fontsize=12)
    # logger.savefig(fig=fig, key=f"figures/score_distribution.png")
    # plt.savefig(fig=fig, fname=f"figures/score_distribution.png")
    fig.show()
    plt.close()
    # %%
    _ = np.array([len(set(row)) for row in traj_labels[top_col_inds]])
    bins = np.arange(1, _.max() + 2, dtype=int)
    fig = plt.figure(figsize=(4, 3))
    plt.title("Out-of-trajectory Neighbors", fontsize=14)
    plt.hist(_, bins=bins, rwidth=0.9, color="#23aaff", alpha=0.7)
    plt.xticks(bins[:-1] + 0.5, bins[:-1])
    plt.ylim(0, 3400)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    logger.savefig(fig=fig, key=f"figures/traj_distribution.png")
    # plt.show()
    # plt.savefig(fig=fig, fname=f"figures/traj_distribution.png")

    logger.log_line(f"## maximum score for top-{k}\n\n{top_cols.max().item()}\n", file=f"top-{k}.md")
    logger.log_line('done', color="green")


@lru_cache(1)
@proto_partial(Config)
def load_pairwise(chunk_prefix, n_chunks):
    from os.path import expanduser
    from tqdm import trange
    from ml_logger import logger
    from multiprocessing.pool import ThreadPool
    import numpy as np

    pool = ThreadPool(min(n_chunks, 5))

    def load(k):
        data, = logger.load_pkl(f"chunk_{k:02d}.pkl")
        print(f'chunk {k} is now loaded!')
        return data

    with logger.PrefixContext(chunk_prefix):
        ds = np.concatenate(pool.map(load, trange(n_chunks, desc="load")))

    rope = np.load(expanduser("~/fair/new_rope_dataset/data/new_rope.npy"), allow_pickle=True)[:10]
    all_images = np.concatenate(rope).transpose(0, 3, 1, 2)

    return ds, all_images


def threshold(**_Config):
    import matplotlib.pyplot as plt
    import numpy as np
    from ml_logger import logger

    Config._update(_Config)

    ds, all_images = load_pairwise()

    assert len(all_images) == len(ds), \
        f"the dimension of the distance matrix {ds.shape} and the images {all_images.shape} should agree."
    logger.log_line('rope dataset is now loaded', color="yellow")

    fig = plt.figure(figsize=(8, 4))
    fig.suptitle('Neighbors Found by Local Metric', fontsize=16)
    for row, query_ind in enumerate(Config.rows):

        img = all_images[query_ind]
        logger.log_image(img[0], "figures/query_image.png")

        neighbor_inds = np.argsort(ds[query_ind], axis=0)

        plt.subplot(4, 11, 11 * row + 1)
        plt.imshow(img[0], cmap='gray')
        if row == 0:
            plt.text(0, -14, "Query", fontsize=10)
        plt.text(0, 60, f"{query_ind}", fontsize=8)
        plt.gca().set_axis_off()

        for i, ind in enumerate(neighbor_inds[:10]):
            plt.subplot(4, 11, 11 * row + i + 2)
            plt.gca().set_axis_off()
            plt.imshow(all_images[ind][0], cmap='gray')
            if row == 0 and i == 0:
                plt.text(0, -14, "Results", fontsize=10)
            score = ds[query_ind, ind]
            plt.text(0, 5, f"{score:0.2f}", fontsize=8, color="red" if score >= Config.threshold else "black")

            plt.text(0, 60, f"{ind}", fontsize=8)

    plt.tight_layout(h_pad=-5)
    logger.savefig(key="figures/neighbor_images.png", dpi=300, bbox_inches='tight')
    plt.close()


def get_topk(ds):
    inds = np.argsort(ds, axis=-1)
    return np.argsort(ds, axis=-1)


@proto_partial(Config)
def build_graph(pairwise, threshold):
    from tqdm import tqdm
    import networkx  as nx
    graph = nx.Graph()

    print(f"threshold is: {threshold}. Average # of neighbors: {(pairwise < threshold).sum() / len(pairwise)}")

    graph.add_nodes_from(range(len(pairwise)))

    inds = np.argsort(pairwise, axis=-1)
    for i, (inds_row, ds_row) in enumerate(tqdm(zip(inds, pairwise), desc="edges: ")):
        ds_sorted = ds_row[inds_row]
        ds_less = ds_sorted[ds_sorted < threshold]
        for j, d in zip(inds_row[:len(ds_less)], ds_less):
            if d < threshold:
                graph.add_edge(i, j, weight=d)

    return graph


# if __name__ == '__main__':
#     from plan2vec_experiments import instr, config_charts
#     from ml_logger import logger
#
#     thunk = instr(load_pairwise)
#     ds, all_images = thunk()
#     graph = build_graph(ds)
#     from graph_search.rope_analysis import plot_graph
#
#     plot_graph(graph)
#     # inds = np.argsort(ds, axis=-1)

if __name__ == "__main__":
    import jaynes

    jaynes.config()

    map_reduce()
    exit()

    from plan2vec_experiments import instr, config_charts

    Config.rows = range(4)  # [100, 500, 800, 1000]
    jaynes.run(instr(threshold), **vars(Config))
    config_charts("""
    charts:
    - type: file
      glob: "**/neighbor_images.png"
    """)

    jaynes.listen()
