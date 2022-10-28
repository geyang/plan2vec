def visualize_neighbors_r(all_states, N, oracle, lng_lat_correction, r=None, path="figures/neighbors.png"):
    """

    :param all_states:
    :param N:
    :param oracle:
    :param lng_lat_correction:
    :param r:
    :param path:
    :return:
    """
    import matplotlib.pyplot as plt
    from ml_logger import logger

    fig = plt.figure(figsize=(5, 5))

    x_inds = [0]
    ns_obs, ns_ds, ns_inds = N(x_inds)
    xs, = oracle(x_inds)
    ns, = oracle(ns_inds)

    magic = [1, lng_lat_correction]
    plt.scatter(*(all_states / magic).T, s=10, color="gray", alpha=0.2)
    plt.scatter(*(ns / magic).T, marker="o", color="#23aaff", alpha=0.1)
    plt.scatter(*(xs / magic).T, marker="x", color="red")
    if r is not None:
        plt.gca().add_artist(plt.Circle(xs / magic, r, color="#23aaff", fill=False, linewidth=4, alpha=0.3))
    plt.gca().set_aspect('equal')
    x_min, y_min = (all_states / magic).min(0)
    x_max, y_max = (all_states / magic).max(0)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    logger.savefig(path)
    fig.show()


colors = ['#49b8ff', '#ff7575', '#66c56c', '#f4b247']


def visualize_neighbors(all_states, x_inds, ns_inds, oracle, lng_lat_correction,
                        r, title=None, path="figures/neighbors.png"):
    """

    :param all_states:
    :param x_inds:
    :param ns_inds:
    :param oracle:
    :param lng_lat_correction:
    :param r:
    :param path:
    :return:
    """
    import matplotlib.pyplot as plt
    from ml_logger import logger

    fig = plt.figure(figsize=(5, 5))

    magic = [1, lng_lat_correction]
    plt.scatter(*(all_states / magic).T, s=10, color="gray", alpha=0.2)

    for i, ns in enumerate(oracle(ns_inds)):
        plt.scatter(*(ns / magic).T, marker="o", color=colors[i % len(colors)], alpha=0.1)
    for i, xs in enumerate(oracle(x_inds)):
        plt.scatter(*(xs / magic).T, marker="x", color="red")
        plt.gca().add_artist(
            plt.Circle(xs / magic, r, color=colors[i % len(colors)], fill=False, linewidth=4, alpha=0.3))

    if title:
        plt.title(title)
    plt.gca().set_aspect('equal')
    x_min, y_min = (all_states / magic).min(0)
    x_max, y_max = (all_states / magic).max(0)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    logger.savefig(path)
    fig.show()

# def visualize_wat(all_states, x_inds, ns_inds, xg_inds, oracle, lng_lat_correction, path="figures/.png"):
#     import matplotlib.pyplot as plt
#     from ml_logger import logger
#
#     fig = plt.figure(figsize=(5, 5))
#
#     magic = [1, lng_lat_correction]
#     plt.scatter(*(all_states / magic).T, s=10, color="gray", alpha=0.2)
#     for i, (xs, ns, xg) in enumerate(zip(oracle(x_inds), oracle(ns_inds), oracle(xg_inds))):
#         plt.scatter(*(ns / magic).T, marker="o", color=colors[i % len(colors)], alpha=0.1)
#         plt.scatter(*(xs / magic).T, marker="x", color=colors[i % len(colors)])
#         plt.scatter(*(xg / magic).T, marker="s", color=colors[i % len(colors)], facecolor="none", linewidths=3)
#     # plt.gca().add_artist(plt.Circle(xs / magic, r, color="#23aaff", fill=False, linewidth=4, alpha=0.3))
#     plt.gca().set_aspect('equal')
#     x_min, y_min = (all_states / magic).min(0)
#     x_max, y_max = (all_states / magic).max(0)
#     plt.xlim(x_min, x_max)
#     plt.ylim(y_min, y_max)
#
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['left'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
#     plt.gca().spines['bottom'].set_visible(False)
#
#     plt.tight_layout()
#     fig.show()
#     logger.savefig(path, dpi=300)


# def visualize_wat(all_states, x_inds, ns_inds, xg_inds, oracle, lng_lat_correction, path="figures/.png"):
#     from tqdm import tqdm
#     import matplotlib.pyplot as plt
#     magic = [1, Args.lng_lat_correction]
#     colors = ['#49b8ff', '#ff7575', '#66c56c', '#f4b247']
#     true_cost = [np.linalg.norm((n_ - g_[None, :]) / magic, ord=2, axis=-1)
#                  for n_, g_ in zip(oracle(ns_inds), oracle(xg_inds))]
#     for i, (c, tc) in enumerate(tqdm(zip(cost, true_cost))):
#         plt.scatter(c, tc, s=20, color=colors[i % len(colors)])
