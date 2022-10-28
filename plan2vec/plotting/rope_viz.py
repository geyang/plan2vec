def border(img, color=0):
    from copy import copy
    c = copy(img)
    c[:, -2:, :] = color
    c[-2:, :, :] = color
    return c


def score_distribution(scores, key=None, *, xlim=None, ylim=None):
    """Visualizes the distribution of scores, w.r.t to a few sample images.

    :param all_samples: Size(N, H, W, C)
    :param query_image: Size(k, H, W, C)
    :param metric_fn: the binary metric function
    :param key: the figure filename
    :return:
    """
    from matplotlib import rcParams
    import matplotlib.pyplot as plt

    rcParams['axes.titlepad'] = 15
    rcParams['axes.labelpad'] = 12
    rcParams['axes.labelsize'] = 14

    # __xs = np.random.rand(1000)
    plt.figure(figsize=(3, 3), dpi=300)
    plt.title('Score Distribution')
    plt.hist(scores, 100, histtype='step', rwidth=0.7, range=xlim)
    plt.yscale('log', nonposy='clip')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)

    if key is None:
        plt.show()
    else:
        from ml_logger import logger
        logger.savefig(key=key)
        plt.close()


def top_neighbors(all_samples, x, scores, key=None):
    """Visualizes the distribution of scores, w.r.t to a few sample images.

    :param all_samples: Size(N, H, W, C), sub-sampled by 10x
    :param x: Size(k, H, W, C) the image we use to query the classifier
    :param scores: The scores from the classifier
    :param key: the figure filename
    :return:
    """
    from ml_logger import logger
    assert len(all_samples) == len(scores), "the length of the samples should be the same as the scores."

    import torch
    with torch.no_grad():
        top_ds, top_inds = torch.topk(scores, 24, dim=-1, largest=False, sorted=True)
        logger.store_metrics(top=top_ds.max().cpu().item())
    _ = all_samples[top_inds.cpu()][:24].cpu().numpy().transpose(0, 2, 3, 1)  # Size(2, 10, 64, 64, 1)
    logger.log_images([border(x.cpu().numpy().transpose(1, 2, 0)), *_], n_rows=5, n_cols=5, key=key)


def faraway_samples(all_samples, x, scores, key=None):
    """Visualizes the distribution of scores, w.r.t to a few sample images.

    :param all_samples: Size(N, H, W, C), sub-sampled by 10x
    :param x: Size(k, H, W, C) the image we use to query the classifier
    :param scores: The scores from the classifier
    :param key: the figure filename
    :return:
    """
    from ml_logger import logger
    assert len(all_samples) == len(scores), "the length of the samples should be the same as the scores."

    import torch
    with torch.no_grad():
        bottom_ds, bottom_inds = torch.topk(scores, 24, dim=-1, largest=True, sorted=True)
        logger.store_metrics(bottom=bottom_ds.max().cpu().item())

    _ = all_samples[bottom_inds.cpu()][:24].cpu().numpy().transpose(0, 2, 3, 1)  # Size(2, 10, 64, 64, 1)
    logger.log_images([border(x.cpu().numpy().transpose(1, 2, 0)), *_], n_rows=5, n_cols=5, key=key)


def visualize_neighbor_states(all_states, xs_inds, ns_inds, filename):
    import matplotlib.pyplot as plt
    from ml_logger import logger

    fig = plt.figure(figsize=(3, 3))
    for row_i in range(len(xs_inds)):
        idx = xs_inds[row_i]
        x0, y0 = all_states[idx]
        for x, y in all_states[ns_inds[row_i]]:
            plt.plot([x0, x], [y0, y], color="black", linewidth=4, alpha=0.4)
    plt.gca().set_aspect('equal')
    plt.title('Neighbors')
    plt.xlim(-0.3, 0.3)
    plt.ylim(-0.3, 0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    logger.savefig(key=filename)
    plt.close(fig)


def visualize_neighbors(xs, ns, xs_inds, prefix):
    """
    useful for both rope dataset and the 2D navigation datasets.

    :param xs: Size(batch_n, *feat)
    :param ns:  Size(batch_n, top_k, *feat)
    :param xs_inds: Tensor(batch_n, dtype=int)
    :param prefix: figure/epoch_{i:04d}
    :return:
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from ml_logger import logger
    from math import ceil

    top_k = ns.shape[1]
    n_rows = ceil(top_k / 6)

    for row_i in range(len(xs)):
        fig = plt.figure(figsize=(3.2, 0.4 + 0.4 * n_rows))
        fig.suptitle('Neighbors')
        gs = GridSpec(n_rows, 7)
        plt.subplot(gs[0, 0])
        plt.title(f'Query', fontsize=6)
        plt.text(5, 85, f'#{xs_inds[row_i]:04d}', fontsize=6)
        plt.imshow(xs[row_i, 0], cmap='gray')
        plt.axis('off')
        for i, neighbor in enumerate(ns[row_i]):
            plt.subplot(gs[i // 6, 1 + i % 6])
            plt.imshow(neighbor[0], cmap='gray')
            plt.axis('off')
        fig.tight_layout(h_pad=-1.6, w_pad=-1.6)
        logger.savefig(f'{prefix}/neighbor_{xs_inds[row_i]:04d}.png')
        plt.close(fig)
