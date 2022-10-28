import numpy as np
import torch
from termcolor import cprint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_grid = None
xys = None


def cache_images(all_images, all_states, lat_correction):
    global image_grid, xys
    if xys is not None:
        cprint('image has already been cached.', 'green')
        return
    xys = np.array(all_states) * [1, lat_correction]
    image_grid = np.array(all_images)


def oracle(img):
    """this is the image grid, simulates a pytorch embedding function"""
    assert img.shape == image_grid.shape, "only works with the original image grid"
    return torch.tensor(xys)


def oracle_3d(img):
    """this is the image grid, simulates a pytorch embedding function"""
    assert img.shape == image_grid.shape, "only works with the original image grid"
    xyzs = np.zeros([xys.shape[0], 3])
    xyzs[:, :2] = xys
    return torch.tensor(xyzs)


def visualize_embedding_2d_image(title, embed_fn, filename=None,
                                 axis_off=False, cmap="gist_rainbow", size=None,
                                 alpha=0.5, figsize=(3.2, 3), dpi=200):
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
    # note: use local import to allow customers to set backend.
    global xys, image_grid
    assert image_grid is not None, "Need to first load the images"
    assert xys is not None, "Need to first load the coordinates"

    from ml_logger import logger

    import torch
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize, dpi=dpi)

    # plt.title(f"{title}", pad=10)
    plt.title(f"{title}", )

    plt.gca().set_aspect('equal', 'datalim')

    images = torch.tensor(image_grid, device=device, dtype=torch.float32)

    logger.print(f"{len(images)} images in dataset", color="green")

    with torch.no_grad():
        zs = embed_fn(images).cpu().numpy()

    plt.scatter(zs[:, 0], zs[:, 1], c=(- xys[:, 0] + xys[:, 1]).flatten(),
                s=size or (6000 / len(images)),
                cmap=cmap, alpha=alpha, linewidths=0)
    plt.xlim(zs[:, 0].min(), zs[:, 0].max())
    plt.ylim(zs[:, 1].min(), zs[:, 1].max())
    if axis_off:
        plt.axis('off')
    else:
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

    plt.tight_layout()

    logger.savefig(filename or f'figures/{title}.png')
    # logger.savefig(filename.replace('png', 'pdf') or f'figures/{title}.pdf')

    plt.close()
    print('Finished visualizing!')


def visualize_embedding_3d_image(title, embed_fn, filename=None):
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
    # note: use local import to allow customers to set backend.
    global xys, image_grid
    assert xys is not None, "Need to first generate the cached images"

    from ml_logger import logger

    import torch
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(3.2, 3), dpi=200)
    ax = Axes3D(fig)

    plt.title(title)

    images = torch.tensor(image_grid, device=device, dtype=torch.float32)

    with torch.no_grad():
        zs = embed_fn(images).cpu().numpy()
    ax.scatter(zs[:, 0], zs[:, 1], zs[:, 2],
               c=(- xys[:, 0] + xys[:, 1]).flatten(), s=6000 / len(images), cmap="gist_rainbow", alpha=0.5,
               linewidths=0)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.grid(False)

    logger.savefig(filename or f'figures/{title}.png')
    # logger.savefig(filename.replace('png', 'pdf') or f'figures/{title}.pdf')
    # uncomment this to visualize locally.
    plt.close()

    print('Finished visualizing!')


if __name__ == "__main__":
    from ml_logger import logger

    for n in [21, 100]:
        title = "simple_shim"
        images = np.zeros([n ** 2, 64, 64, 3])
        xs = np.linspace(-1, 1, n)
        xs, ys = np.meshgrid(xs, xs)

        cache_images(images, np.stack([xs, ys]).T.reshape(-1, 2), 0.75)
        # plotting the oracle
        visualize_embedding_2d_image(title, oracle, f"figures/{n:04d}-{title}_embedding_oracle.png")
        visualize_embedding_3d_image(title, oracle_3d, f"figures/{n:04d}-{title}_embedding_oracle_3d.png")
