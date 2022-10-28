from os.path import exists, expanduser

import numpy as np
import torch
from matplotlib.font_manager import FontProperties
from termcolor import cprint

GOALS = {
    'CMazeDiscreteImgIdLess-v0': np.array([-0.15, -0.15]),
    'GoalMassDiscreteImgIdLess-v0': np.array([0.15, -0.15]),
    '回MazeDiscreteImgIdLess-v0': np.array([0.15, -0.05]),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_grid = None
xys = None
indices = None
grid_n = None

goal_image = None
xy = None


def cache_images(env_id, n=21, width=64, height=64):
    global image_grid, xys, goal_image, xy, indices, grid_n
    from ge_world import gym
    from tqdm import tqdm
    from ml_logger import logger

    if xys is not None:
        cprint('image has already been generated.', 'green')
        return

    grid_n = n
    xys, image_grid, indices = [], [], []
    env = gym.make(env_id)
    # reverse the y order to go from top to bottom.
    for i, y in enumerate(tqdm(np.linspace(env.obj_low, env.obj_high, n)[::-1], desc="caching images...")):
        for j, x in enumerate(np.linspace(env.obj_low, env.obj_high, n)):
            _ = (x, y)
            if env.is_good_goal(_):
                env.reset_model(x=_)
                image = env.unwrapped._get_obs()['img']
                image_grid.append(image)
                indices.append([i, j])
                xys.append(_)

    xys = np.array(xys)
    image_grid = np.array(image_grid)

    xy = GOALS[env_id]
    env.reset_model(x=xy)
    goal_image = env.unwrapped._get_obs()['img']
    env.close()

    # save the image for debugging purposes
    logger.log_video(image_grid[:, 0, ...], key="debug/goal_samples.mp4", fps=30)
    logger.log_image(goal_image[0], key="debug/goal_image.png")


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
                                 axis_off=False, cmap="gist_rainbow",
                                 alpha=0.5, figsize=(3.2, 3), dpi=None):
    """

    :param title: the title of your plot. If no figure filename is specified, this is used as the figure name.
    :param embed_fn: the embedding function, typically your neural network.
    :param filename: Default to f"figures/{title}.png". If you want to overwrite this make
        sure you use some ./figures/ namespace.
    :return:
    """
    from ml_logger import logger

    # note: use local import to allow customers to set backend.
    global xys, image_grid
    assert xys is not None, "Need to first generate the cached images"

    import torch
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize, dpi=dpi)

    # download CJK font from https://noto-website-2.storage.googleapis.com/pkgs/Noto-unhinted.zip
    CJK_font_path = expanduser('~/opentype/noto/NotoSansCJKsc-Regular.otf')
    if exists(CJK_font_path):
        plt.title(f"{title}", fontproperties=FontProperties(fname=CJK_font_path))
    else:
        cprint(f"Please Install noto font, and place at {CJK_font_path}.", "red")
        cprint(f"see https://noto-website-2.storage.googleapis.com/pkgs/Noto-unhinted.zip", "green")
        plt.title(f"{title}", )

    plt.gca().set_aspect('equal', 'datalim')

    images = torch.tensor(image_grid, device=device, dtype=torch.float32)

    with torch.no_grad():
        zs = embed_fn(images).cpu().numpy()

    if False:  # this is to normalize small clusters.
        zs = (zs - zs.mean(axis=0)) / (zs.max(axis=0) - zs.min(axis=0))

    plt.scatter(zs[:, 0], zs[:, 1], c=(- xys[:, 0] + xys[:, 1]).flatten(), cmap=cmap, alpha=alpha,
                linewidths=0)

    if axis_off:
        plt.axis('off')
    else:
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

    logger.savefig(filename or f'figures/{title}.png')
    # logger.savefig(filename.replace('png', 'pdf') or f'figures/{title}.pdf')
    plt.close()

    print('Finished visualizing!')

    return {"z_x": zs[:, 0], "z_y": zs[:, 1],
            "color": (- xys[:, 0] + xys[:, 1]).flatten(),
            "x": xys[:, 0], "y": xys[:, 1]}


def visualize_embedding_3d_image(title, embed_fn, filename=None, axis_off=False, cmap="gist_rainbow",
                                 alpha=0.5, figsize=(3.2, 3), dpi=None, show=False):
    """

    :param title: the title of your plot. If no figure filename is specified, this is used as the figure name.
    :param embed_fn: the embedding function, typically your neural network.
    :param filename: Default to f"figures/{title}.png". If you want to overwrite this make
        sure you use some ./figures/ namespace.
    :return:
    """
    from ml_logger import logger

    # note: use local import to allow customers to set backend.
    global xys, image_grid
    assert xys is not None, "Need to first generate the cached images"

    import torch
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d, Axes3D

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = Axes3D(fig)
    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 1, 0.75]))

    # download CJK font from https://noto-website-2.storage.googleapis.com/pkgs/Noto-unhinted.zip
    CJK_font_path = expanduser('~/opentype/noto/NotoSansCJKsc-Regular.otf')
    if exists(CJK_font_path):
        plt.title(f"{title}", fontproperties=FontProperties(fname=CJK_font_path))
    else:
        cprint(f"Please Install noto font, and place at {CJK_font_path}.", "red")
        cprint(f"see https://noto-website-2.storage.googleapis.com/pkgs/Noto-unhinted.zip", "green")
        ax.set_title(f"{title}", pad=-1)

    images = torch.tensor(image_grid, device=device, dtype=torch.float32)

    with torch.no_grad():
        zs = embed_fn(images).cpu().numpy()

    ax.scatter(zs[:, 0], zs[:, 1], zs[:, 2],
               c=(- xys[:, 0] + xys[:, 1]).flatten(), cmap=cmap, alpha=alpha, linewidths=0)

    if axis_off:
        plt.axis('off')
    else:
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.grid(False)

    logger.savefig(filename or f'figures/{title}.png')
    # logger.savefig(filename.replace('png', 'pdf') or f'figures/{title}.pdf')
    # uncomment this to visualize locally.
    if show:
        plt.show()
    plt.close()

    print('Finished visualizing!')

    return {"z_x": zs[:, 0], "z_y": zs[:, 1], "z_z": zs[:, 2],
            "color": (- xys[:, 0] + xys[:, 1]).flatten(),
            "x": xys[:, 0], "y": xys[:, 1]}


def visualize_value_map(value_fn, filename):
    from ml_logger import logger

    # note: use local import to allow customers to set backend.
    global xys, image_grid, xy, goal_image, grid_n, indices
    assert xys is not None, "Need to first generate the cached images"

    import torch

    images = torch.tensor(image_grid, device=device, dtype=torch.float32)

    zs = np.zeros([grid_n, grid_n])

    with torch.no_grad():
        _ = value_fn(images, goal_image[None, ...]).cpu().numpy()
        for (i, j), v in zip(indices, _):
            zs[i, j] = v

    logger.log_image(zs, key=filename, cmap='RdBu', normalize='individual')

    print('Finished visualizing the value function')


if __name__ == "__main__":

    for title, env_id in [
        ("Goal-Mass", "GoalMassDiscreteImgIdLess-v0"),
        ("回-Maze", "回MazeDiscreteImgIdLess-v0",),
        ("C-Maze", "CMazeDiscreteImgIdLess-v0")
    ]:
        cache_images(env_id, n=21)
        # plotting the oracle
        visualize_embedding_2d_image(title, oracle, f"figures/{title}_embedding_oracle.png")
        visualize_embedding_3d_image(title, oracle_3d, f"figures/{title}_embedding_oracle_3d.png")

    # debug the inputs
    # logger.log_images(image_grid.transpose(0, 2, 3, 1), n_cols=21, n_rows=21, key='figures/embedding_visual.png')
