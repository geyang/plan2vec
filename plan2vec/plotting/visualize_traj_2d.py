import numpy as np
from tqdm import tqdm


def visualize_prediction(s, s_prime, labels, k=10, *, is_positive=lambda y: y > 0.5, key=None, title=None):
    """
    Visualizing the sample trajectories in a 2-dimensional domain

    :param s: the dictionary of the sampled dataset
    :param s_prime:
    :param labels: the predicted label between 0 and 1
    :param k: the number of sub-samples
    :param key: The path to save the figure to
    :param title: Optional Title for the plot
    :return:
    """
    import matplotlib.pyplot as plt
    from ml_logger import logger

    DPI = 300
    title = title or "Pairs"

    plt.figure(figsize=(3, 3), dpi=DPI, )
    plt.title(title)

    selected = np.random.rand(len(s)).argsort()[:k]
    trajs = np.concatenate([s[selected][:, None, :], s_prime[selected][:, None, :]], axis=1)
    ys = labels[selected]

    k = trajs.shape[0]

    # done: ues different color for different trajectories
    # note: marker size = points/inch * actual axial size.
    for i, (traj, y) in enumerate(tqdm(zip(trajs, ys))):
        plt.plot(traj[:, 0], traj[:, 1], 'o-', c="#23aaff" if is_positive(y) else "red", alpha=0.5,
                 linewidth=2, markersize=DPI * 0.015, markeredgecolor="none")

    plt.xlim(-0.3, 0.3)
    plt.ylim(-0.3, 0.3)
    plt.gca().set_aspect('equal')

    if key is None:
        plt.show()
    else:
        logger.savefig(key)
    plt.close()


def visualize_eval_prediction(s, s_prime, pred, labels, k=10, *, is_positive=lambda y: y > 0.5 and y < 1.5,
                              identity=lambda y: y <= 0.5, correct=None, key=None, title=None):
    """
    Visualizing the sample trajectories in a 2-dimensional domain

    :param s: the dictionary of the sampled dataset
    :param s_prime:
    :param labels: the predicted label between 0 and 1
    :param k: the number of sub-samples
    :param key: The path to save the figure to
    :param title: Optional Title for the plot
    :return:
    """
    import matplotlib.pyplot as plt
    from ml_logger import logger
    DPI = 300
    title = title or "Pairs"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2), dpi=DPI)
    ax2.set_title("Ground Truth")
    ax1.set_title("Prediction")

    selected = np.random.rand(len(s)).argsort()[:k]
    trajs = np.concatenate([s[selected][:, None, :], s_prime[selected][:, None, :]], axis=1)
    ys = labels[selected]
    yhats = pred[selected]

    k = trajs.shape[0]

    # done: ues different color for different trajectories
    # note: marker size = points/inch * actual axial size.
    for i, (traj, y, yhat) in enumerate(tqdm(zip(trajs, ys, yhats))):
        if is_positive(y):
            color = "#23aaff"
        elif identity(y):
            color = "green"
        else:
            color = "red"
        ax2.plot(traj[:, 0], traj[:, 1], 'o-', c=color, alpha=0.5,
                 linewidth=2, markersize=DPI * 0.015, markeredgecolor="none")

        if is_positive(yhat):
            color = "#23aaff"
        elif identity(yhat):
            color = "green"
        else:
            color = "red"
        if not correct[i]:
            color = "black"
        ax1.plot(traj[:, 0], traj[:, 1], 'o-', c=color, alpha=0.5,
                 linewidth=2, markersize=DPI * 0.015, markeredgecolor="none")

    ax1.set_xlim(-0.3, 0.3)
    ax1.set_ylim(-0.3, 0.3)
    ax2.set_xlim(-0.3, 0.3)
    ax2.set_ylim(-0.3, 0.3)
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    fig.tight_layout()

    if key is None:
        plt.show()
    else:
        logger.savefig(key, bbox_inches="tight")
    plt.close()


def local_metric_vs_ground_truth(s, s_prime, y_hat, k=100, yDomain=(-0.05, 1.05), key=None):
    """
    Visualizing the sample trajectories in a 2-dimensional domain

    :param s: first observation
    :param s_prime: second observation
    :param y_hat: The output of the network
    :param key: The path to save the figure to
    :return:
    """
    import matplotlib.pyplot as plt
    from ml_logger import logger
    import pandas as pd

    DPI = 300
    title = "Local Metric vs Ground Truth"

    plt.figure(figsize=(3, 2), dpi=DPI, )
    plt.title(title)

    plt.ylabel("Score")
    plt.xlabel("L1 Distance")

    d_l1 = np.abs(s_prime - s).sum(axis=-1)
    df = pd.DataFrame(dict(d=d_l1, y=y_hat))
    out, bins = pd.qcut(df['d'], k, retbins=True, duplicates='drop')
    grouped = df.groupby(out)
    new_df = pd.merge(grouped['y'].agg(['count', 'mean', 'min', 'max']).reset_index(),
                      grouped['y'].describe(percentiles=[0.25, 0.75, 0.5, 0.05, 0.95]))

    color = "#23aaff"
    plt.plot(new_df['d'].apply(lambda d: d.left).values.tolist(),
             new_df['50%'].values.tolist(),  # use the median instead of the mean for quantile range
             color=color,
             linewidth=2, markersize=DPI * 0.015, markeredgecolor="none")
    plt.fill_between(new_df['d'].apply(lambda d: d.left).values.tolist(),
                     new_df['25%'].values.tolist(),
                     new_df['75%'].values.tolist(),
                     color=color, alpha=0.2, linewidth=0)
    plt.ylim(*yDomain)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    if key is None:
        plt.show()
    else:
        logger.savefig(key, bbox_inches='tight')
    plt.close()


def local_metric_over_trajectory(y_hat, key=None, *, ylim=(-0.05, 1.05), xlim=None):
    """Visualizing the sample trajectories in a 2-dimensional domain

    :param y_hat: Size(Batch, Timesteps, 1), the score for the trajectory
    :param key: file key to which the figure is saved.
    :param ylim: vertical limits to the plot
    :return:
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from ml_logger import logger

    from matplotlib import rcParams
    rcParams['axes.titlepad'] = 15
    rcParams['axes.labelpad'] = 12
    rcParams['axes.labelsize'] = 12

    DPI = 300
    title = "Local Metric Over Trajectory"

    plt.figure(figsize=(3, 2), dpi=DPI, )
    plt.title(title)

    plt.ylabel("Score")
    plt.xlabel("Time Steps Apart")

    sns.tsplot(y_hat, time=range(y_hat.shape[1]), ci=[75, 100], color="#23aaff")
    plt.ylim(*ylim)
    if xlim:
        plt.xlim(*xlim)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    if key is None:
        plt.show()
    else:
        logger.savefig(key, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    pass
