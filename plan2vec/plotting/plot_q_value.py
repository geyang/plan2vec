import numpy as np
import warnings
from ml_logger import logger


def visualize_q_2d(q_fn, low=-.25, high=0.25, n=21, goal_n=3, cmap=None, title=None, key=None):
    """
    we sample the position of the robot and the goal, and plot the Q-function

    :param title: the title of your plot. If no figure filename is specified, this is used as the figure name.
    :param q_fn: the goal-conditioned Q-function, with signature (x, g) -> v
    :param low: the lower end of the range of x, y
    :param high: the higher end of the range of x, y
    :param n: number of bins for x, and y. should always be odd.
    :param goal_n: the number of bins for the goal
    :param act: the action to plot. Assumes that Q function outputs in Size[act_dim]
    :param filename: Default to f"figures/{title}.png". If you want to overwrite this make
        sure you use some ./figures/ namespace.
    :return:
    """
    assert low < high, f"the lower limit ({low}) need to be less than than the higher range ({high})."

    if n % 2 == 0:
        warnings.warn(f"Are you sure that you don't want n ({n}) to be odd? ")

    xs = ys = np.linspace(low, high, n)
    xs, ys = np.meshgrid(xs, ys)

    _ = np.concatenate([xs[:, :, None], ys[:, :, None]], axis=-1).reshape(-1, 2)

    g_ = np.linspace(low, high, goal_n)
    g_xs, g_ys = np.meshgrid(g_, g_)

    value_map = [-q_fn(_, [[x, y]] * _.shape[0]).reshape(n, n) for x, y in zip(g_xs.flatten(), g_ys.flatten())]
    # todo: The problem of using log_images is that we can not add text.
    key = key or f"figures/{title}.png"
    logger.log_images(value_map, key, n_cols=goal_n, n_rows=goal_n, cmap=cmap or 'RdBu',
                      normalize='individual')


if __name__ == "__main__":
    l2_Q = lambda xys, goal_xys: np.tile(- np.linalg.norm(goal_xys - xys, ord=2, axis=-1)[:, None], 9)

    logger.configure(register_experiment=False)
    visualize_q_2d(q_fn=l2_Q, cmap='RdBu', title='Cartesian Distance', key='figures/Cartesian Distance.png')
