import numpy as np
import warnings


# brutal = c
import torch


def visualize_embedding_2d(title, embed_fn, low=-.25, high=0.25, n=21, filename=None):
    """

    :param title: the title of your plot. If no figure filename is specified, this is used as the figure name.
    :param embed_fn: the embedding function, typically your neural network.
    :param low: the lower end of the range of x, y
    :param high: the higher end of the range of x, y
    :param n: number of bins for x, and y. should always be odd.
    :param filename: Default to f"figures/{title}.png". If you want to overwrite this make
        sure you use some ./figures/ namespace.
    :return:
    """
    from ml_logger import logger

    # note: use local import to allow customers to set backend.
    import matplotlib.pyplot as plt
    assert low < high, f"the lower limit ({low}) need to be less than than the higher range ({high})."

    plt.figure(figsize=(3.2, 3))

    if n % 2 == 0:
        warnings.warn(f"Are you sure that you don't want n ({n}) to be odd? ")

    xs = ys = np.linspace(low, high, n)
    xs, ys = np.meshgrid(xs, ys)

    plt.title(title)

    plt.gca().set_aspect('equal', 'datalim')

    _ = np.concatenate([xs[:, :, None], ys[:, :, None]], axis=-1).reshape(-1, 2)
    with torch.no_grad():
        zs = embed_fn(_)
    plt.scatter(zs[:, 0], zs[:, 1], c=(- xs + ys).flatten(), cmap="gist_rainbow", alpha=0.5, linewidths=0)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    logger.savefig(filename or f'figures/{title}.png')
    # uncomment this to visualize locally.
    plt.close()

    print('Finished visualizing!')

    # return data to be saved.
    return {"z_x": zs[:, 0], "z_y": zs[:, 1],
            "color": (- xs[:, 0] + ys[:, 1]).flatten(),
            "x": xs[:, 0], "y": ys[:, 1]}


def visualize_embedding_3d(title, embed_fn, low=-.25, high=0.25, n=21, filename=None):
    """

    :param title: the title of your plot. If no figure filename is specified, this is used as the figure name.
    :param embed_fn: the embedding function, typically your neural network.
    :param low: the lower end of the range of x, y
    :param high: the higher end of the range of x, y
    :param n: number of bins for x, and y. should always be odd.
    :param filename: Default to f"figures/{title}.png". If you want to overwrite this make
        sure you use some ./figures/ namespace.
    :return:
    """
    from ml_logger import logger

    # note: use local import to allow customers to set backend.
    import matplotlib.pyplot as plt
    assert low < high, f"the lower limit ({low}) need to be less than than the higher range ({high})."
    from mpl_toolkits.mplot3d import axes3d, Axes3D

    fig = plt.figure(figsize=(3.2, 3))
    ax = Axes3D(fig)

    plt.figure(figsize=(3.2, 3))

    if n % 2 == 0:
        warnings.warn(f"Are you sure that you don't want n ({n}) to be odd? ")

    xs = ys = np.linspace(low, high, n)
    xs, ys = np.meshgrid(xs, ys)

    plt.title(title)

    plt.gca().set_aspect('equal', 'datalim')

    _ = np.concatenate([xs[:, :, None], ys[:, :, None]], axis=-1).reshape(-1, 2)
    with torch.no_grad():
        zs = embed_fn(_)
    ax.scatter(zs[:, 0], zs[:, 1], c=(- xs + ys).flatten(), cmap="gist_rainbow", alpha=0.5, linewidths=0)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.grid(False)

    logger.savefig(filename or f'figures/{title}.png')
    logger.savefig(filename or f'figures/{title}.pdf')
    # uncomment this to visualize locally.
    plt.close()

    print('Finished visualizing!')

    # return data to be saved.
    return {"z_x": zs[:, 0], "z_y": zs[:, 1], "z_z": zs[:, 2],
            "color": (- xs[:, 0] + ys[:, 1]).flatten(),
            "x": xs[:, 0], "y": ys[:, 1]}


identity_fn = lambda xys: xys


def get_ball_fn(order=0.1):
    """factory for ball functions"""

    def ball_fn(xys):
        rms_norm = (xys[:, 0] ** 2 + xys[:, 1] ** 2 + np.finfo(float).eps) ** order
        return xys / rms_norm[:, None]

    ball_fn.__name__ = f"ball({order:0.2f})"

    return ball_fn


def get_model_fn(model, action_space, latent_dim, name, device):
    import torch

    def model_fn(xys):
        value_xys = model(torch.Tensor(xys).to(device)).view(
            -1, action_space, latent_dim)[:, 4].detach().cpu()
        return value_xys

    model_fn.__name__ = f"model({name})"

    return model_fn


def get_model_sep_fn(model, action_space, latent_dim, name, device):
    import torch

    def model_fn(xys):
        value_xys = model.encode(torch.Tensor(xys).to(device)).detach().cpu()
        return value_xys

    model_fn.__name__ = f"model({name})"

    return model_fn


if __name__ == "__main__":
    visualize_embedding_2d(title='Point Mass (State)', embed_fn=identity_fn)

    _ = get_ball_fn()
    visualize_embedding_2d(title=f'Point Mass, {_.__name__}', embed_fn=_)

    _ = get_ball_fn(-0.1)
    visualize_embedding_2d(title=f'Point Mass, {_.__name__}', embed_fn=_)
