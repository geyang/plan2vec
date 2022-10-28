from collections import defaultdict
from functools import partial

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from params_proto.neo_proto import ParamsProto

from graph_search import methods, short_names
from plan2vec_experiments.analysis_icml_2020 import mlc, stylize, plot_bar
from streetlearn import StreetLearnDataset


class Args(ParamsProto):
    env_id = "streetlearn_small"
    neighbor_r = 2.4e-4
    neighbor_r_min = None

    h_scale = 1.2

    # plotting
    visualize_graph = True


def load_streetlearn(
        data_path="~/fair/streetlearn/processed-data/manhattan-large",
        pad=0.1,
        visualize=True,
        view_size=None,
        view_mode=None,
):
    from streetlearn import StreetLearnDataset
    import matplotlib.pyplot as plt
    from os.path import expanduser
    path = expanduser(data_path)

    d = StreetLearnDataset(path, view_size=view_size, view_mode=view_mode)
    d.select_bbox(-73.997, 40.726, 0.01, 0.008)

    if visualize:
        d.show_blowout("NYC-large", show=visualize)

    a = d.bbox[0] + d.bbox[2] * pad, d.bbox[1] + d.bbox[3] * pad
    b = d.bbox[0] + d.bbox[2] * (1 - pad), d.bbox[1] + d.bbox[3] * (1 - pad)
    (start, _), (goal, _) = d.locate_closest(*a), d.locate_closest(*b)

    if visualize:
        fig = plt.figure(figsize=(6, 5))
        plt.scatter(*d.lng_lat[start], marker="o", s=100, linewidth=3,
                    edgecolor="black", facecolor='none', label="start")
        plt.scatter(*d.lng_lat[goal], marker="x", s=100, linewidth=3,
                    edgecolor="none", facecolor='red', label="end")
        plt.legend(loc="upper left", bbox_to_anchor=(0.95, 0.7), framealpha=1,
                   frameon=False, fontsize=12)
        d.show_blowout("NYC-large", fig=fig, box_color='gray', box_alpha=0.1,
                       show=True, set_lim=True)

    return d, start, goal

    # 1. get data
    # 2. build graph
    # 3. get start and goal
    # 4. make plans


def plot_graph(graph):
    # fig = plt.figure(figsize=(3, 3))
    nx.draw(graph, [n['pos'] for n in graph.nodes.values()],
            node_size=0, node_color="gray", alpha=0.7, edge_color="gray")
    plt.gca().set_aspect('equal')
    # plt.tight_layout()
    # plt.show()


def maze_graph(dataset: StreetLearnDataset):
    from tqdm import tqdm

    all_nodes = dataset.lng_lat
    graph = nx.Graph()
    for node, xy in enumerate(tqdm(all_nodes, desc="build graph")):
        graph.add_node(node, pos=xy)

    for node, a in tqdm(graph.nodes.items(), desc="add edges"):
        (ll,), (ds,), (ns,) = dataset.neighbor([node], r=Args.neighbor_r)
        for neighbor, d in zip(ns, ds):
            graph.add_edge(node, neighbor, weight=d)

    return graph
    # if Args.visualize_graph:
    #     plot_graph(graph)
    #     plt.gca().set_aspect(dataset.lat_correction)
    #     plt.show()


# noinspection PyPep8Naming,PyShadowingNames
def heuristic(a, b, G: nx.Graph, scale=1, lat_correction=1 / 0.74):
    a = [G.nodes[n]['pos'] for n in a]
    b = [G.nodes[n]['pos'] for n in b]
    magic = [1, lat_correction]
    return np.linalg.norm((np.array(a) - np.array(b)) * magic, ord=1, axis=-1) * scale


def plot_trajectory_2d(path, color='black', **kwargs):
    for (x, y), (x_, y_) in zip(path[:-1], path[1:]):
        dx = (x_ - x)
        dy = (y_ - y)
        d = np.linalg.norm([dx, dy], ord=2)
        plt.arrow(x, y, dx * 0.8, dy * 0.8, **kwargs, head_width=d * 0.3, head_length=d * 0.3,
                  length_includes_head=True, head_starts_at_zero=True, fc=color, ec=color)


def set_fig(dataset: StreetLearnDataset):
    plt.gca().set_yticklabels([])
    plt.gca().set_xticklabels([])
    plt.gca().set_aspect(dataset.lat_correction)


def ind2pos(G, inds, scale=1):
    return [G.nodes[n]['pos'] * scale for n in inds]


def patch_graph(G):
    queries = defaultdict(lambda: 0)
    _neighbors = G.neighbors

    def neighbors(n):
        # queries[n] += 1  # no global needed bc mutable.
        ns = list(_neighbors(n))
        for n in ns:
            queries[n] += 1
        return ns

    G.neighbors = neighbors
    return queries


@mlc
def main():
    from waterbear import DefaultBear
    from ml_logger import logger

    dataset, start, goal = load_streetlearn()
    G = maze_graph(dataset)
    queries = patch_graph(G)

    # goal -= 120 # 10 worked well
    cache = DefaultBear(dict)

    titles = dict(a_star="A* (L1)")

    for i, (key, search) in enumerate(methods.items()):
        queries.clear()
        title, *_ = search.__doc__.split('\n')
        short_name = short_names[key]

        path, ds = search(G, start, goal, partial(heuristic, G=G, scale=1.2))
        cache.cost[short_name] = len(queries.keys())
        cache.len[short_name] = len(ds)
        logger.print(f"{key:>10} len: {len(path)}", f"cost: {len(queries.keys())}")
        # plt.subplot(2, 2, i + 1)
        # plt.figure(figsize=(4, 4), dpi=300)
        plt.figure(figsize=(2.1, 2.4), dpi=300)
        plt.title(titles[key] if key in titles else title, pad=10)
        plot_trajectory_2d(ind2pos(G, path, 100), label=short_name)
        plt.scatter(*zip(*ind2pos(G, queries.keys(), 100)), color="gray", s=3, alpha=0.1)
        set_fig(dataset)

        plt.tight_layout()
        logger.savefig(f"figures/sl_plans_{key}.png", dpi=300)
        plt.show()
        plt.close()


@mlc
def plan2vec(checkpoint, latent_dim=2):
    from waterbear import DefaultBear
    from ml_logger import logger

    logger.print(f'Loading data from')
    logger.print(checkpoint)

    dataset, start, goal = load_streetlearn(
        view_size=(64, 64), view_mode="omni_gray", visualize=False)
    G = maze_graph(dataset)
    queries = patch_graph(G)

    all_images = dataset.images[:, None, ...] / 255

    from plan2vec.models.resnet import ResNet18CoordL2, ResNet18L2, ResNet18Kernel

    # does not support cuda
    Φ = ResNet18L2(1, latent_dim=latent_dim, p=2)
    import torch
    from torch_utils import Eval, torchify
    from ml_logger import ML_Logger
    from plan2vec_experiments import RUN
    load_logger = ML_Logger(RUN.server)
    load_logger.load_module(Φ, checkpoint)
    del load_logger

    @torchify(dtype=torch.float32)
    def d_eval(*_):
        with Eval(Φ), torch.no_grad():
            return Φ(*_).squeeze()

    def img_eval(a, b):
        return d_eval(all_images[a], all_images[b])

    # goal -= 120 # 10 worked well
    cache = DefaultBear(dict)

    _ = [["a_star", "Plan2Vec*", "plan2vec_star"], ["heuristic", "Plan2Vec", 'plan2vec']]

    for i, (key, title, file_key) in enumerate(_):
        search = methods[key]

        queries.clear()
        # title, *_ = search.__doc__.split('\n')
        short_name = short_names[key]

        path, ds = search(G, start, goal, img_eval)
        cache.cost[short_name] = len(queries.keys())
        cache.len[short_name] = len(ds)
        logger.print(f"{title:>10} len: {len(path)}", f"cost: {len(queries.keys())}")
        # plt.subplot(2, 2, i + 1)
        # plt.figure(figsize=(4, 4), dpi=300)
        plt.figure(figsize=(2.1, 2.4), dpi=300)
        plt.title(title, pad=10)
        plot_trajectory_2d(ind2pos(G, path, 100), label=short_name)
        plt.scatter(*zip(*ind2pos(G, queries.keys(), 100)), color="red", s=3, alpha=0.3)
        set_fig(dataset)

        plt.tight_layout()
        logger.savefig(f"figures/sl_plans_{file_key}.png", dpi=300)
        plt.show()
        plt.close()


@mlc
def make_sl_bar_chart():
    """
    # Data:

     UVPN* len: 125 cost: 200
      UVPN len: 125 cost: 200
       bfs len: 123 cost: 1470
 heuristic len: 124 cost: 195
  dijkstra len: 123 cost: 1470
    a_star len: 124 cost: 342
    :return:
    """
    import pandas as pd
    data = pd.DataFrame([
        # dict(name="UVPN*", len=125, cost=200, ),
        dict(name="Plan2Vec*", len=125, cost=200, ),
        # dict(name="bfs", len=123, cost=1470, ),
        # dict(name="heuristic (L1)", len=124, cost=195, ),
        dict(name="A* (L1)", len=124, cost=342, ),
        dict(name="Dijkstra's", len=123, cost=1470, )]).set_index("name")

    plot_bar(data.index, data['cost'], title="Planning Cost",
             labels=[f"{l:0.1f}" for l in data['cost'] / 125],
             figsize=(2.2, 2.5),
             ylabel="# of Expansion",
             # ylim=(0, 1500),
             xticks_rotation=40,
             filename="figures/sl_cost_comparison.pdf")

    # plot_bar(data.index, data['cost'] / 125,
    #          labels=data['cost'],
    #          title="Cost per Step",
    #          figsize=(2.4, 2.4),
    #          # ylim=(0, 1500),
    #          ylabel="Expansion per Step",
    #          xticks_rotation=40,
    #          yticks=dict(ticks=[0, 1, 2, 3, 10, 11, 12, 13]),
    #          key="figures/cost_comparison_sl_normalized.png")


if __name__ == '__main__':
    stylize()
    main()
    plan2vec(
        checkpoint="/amyzhang/plan2vec/2020/02-02/analysis_icml_2020/train/"
                   "streetlearn_uvpn/sweep-L_p/16.19/lr(3e-06)-rs-(2000)-p2/"
                   "manhattan-large/21.149056/models/04000/Φ.pkl",
    )
    make_sl_bar_chart()
