from functools import wraps
from os.path import expanduser
from matplotlib.colors import LinearSegmentedColormap

COLORS = ['#49b8ff', '#ff7575', '#66c56c', '#f4b247']
LINE_STYLES = ['-', '-.', '--', ':']
HUES = [
    "#fc3f8c",
    "#ee2151",
    "#d4191f",
    "#e04408",
    "#fd8609",
    "#f4c309",
    "#a3bb07",
    "#3f910f",
    "#1baa54",
    "#1fedc9",
    "#1cdeed",
    "#15a4d6",
    "#2266af",
    "#7c57b1",
    "#ca4eb0"
]
cmap = LinearSegmentedColormap.from_list("ge", HUES)

# LOG_DIR = "~/Dropbox/Apps/ShareLaTeX/Plan2Vec Unsupervised Representation " \
#           "Learning By Latent Plans/ICML_2020_submission/06_experiments"
# LOG_DIR = "~/Dropbox/Apps/ShareLaTeX/Universal Value Prediction Network/UVPN_ICML_2020" \
#           "/06_experiments"
LOG_DIR = "~/Dropbox/Apps/ShareLaTeX/Plan2vec Camera Ready/experiments"


def mlc(func):
    """ML Logger configuration decorator"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        from ml_logger import logger
        logger.configure(log_directory=expanduser(LOG_DIR), prefix="", register_experiment=False)
        return func(*args, **kwargs)

    return wrapper


def stylize():
    import matplotlib.pyplot as plt
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')


@mlc
def plot_bar(keys, values, stds=None, labels=None, title=None,
             figsize=(2.4, 2.4),
             xlabel=None, ylabel=None, ylim=None, ylim_scale=1.2,
             xticks_rotation=None, yticks=None, filename=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize, dpi=300)

    if title:
        plt.title(title, pad=10)

    means = [v.mean() if hasattr(v, "mean") else v for v in values]
    stds = stds or [v.std() for v in values if hasattr(v, "std")] or None

    plt.bar(keys, means, yerr=stds, color="gray", width=0.8)

    if ylim:
        plt.ylim(*ylim)
    else:
        plt.ylim(0, max(means) * ylim_scale)

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    if xticks_rotation:
        plt.xticks(rotation=xticks_rotation)

    if yticks:
        plt.yticks(**yticks)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.tight_layout()

    if labels is not None:
        rects = plt.gca().patches

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            plt.gca().text(rect.get_x() + rect.get_width() / 2, height + 0.25, label,
                           ha='center', va='bottom')

    if filename:
        from ml_logger import logger
        logger.savefig(filename)

    plt.show()


@mlc
def plot_line(data_frame, xKey, *yKeys, title=None, raw=False, smooth=5,
              var_window=None, color=None,
              colors=None,
              styles=None,
              figsize=(3.8, 2.4),
              legend=dict(bbox_to_anchor=(0.95, 0.8), fontsize=10),
              xlabel=None, ylabel=None, ynorm=False, ypc=False,
              xlim=None, ylim=None,
              xticks=None,
              filename=None):
    import matplotlib.pyplot as plt

    colors = colors or COLORS
    styles = styles or LINE_STYLES

    plt.figure(figsize=figsize)

    if title:
        plt.title(title, pad=15)
    if ynorm:
        data_frame = data_frame.copy()
        for i, yKey in enumerate(yKeys):
            ys = data_frame[yKey]
            data_frame[yKey] = ys / ys.rolling(20).mean().max()

    l = len(yKeys)
    for i, yKey in enumerate(yKeys):
        s = data_frame[[xKey, yKey]].dropna()

        label = yKey.replace('_', " ")
        xs, ys = s[xKey], s[yKey]
        if raw:
            plt.plot(xs, ys, color=color if color else 'grey', alpha=0.7, linewidth=0.1,
                     label=None if smooth else label)

        if smooth:
            var_window = var_window or smooth // 10
            if smooth < 1:
                rolling = ys.ewm(alpha=smooth)
            else:
                rolling = ys.rolling(smooth, min_periods=1)
            smoothed = rolling.mean()
            std = ys.rolling(var_window or 1, min_periods=1).mean().rolling(10, min_periods=1).std()
            plt.plot(xs, smoothed, color=color if color else colors[i % 4],
                     linestyle=styles[i % 4], label=label, zorder=l * 2 - i * 2 + 1)

            p, m = smoothed + std, smoothed - std
            plt.fill_between(xs, y1=m, y2=p, alpha=0.3, color=colors[i % 4], zorder=l * 2 - i * 2)
        else:
            plt.plot(xs, ys, color=color if color else colors[i % 4],
                     linestyle=styles[i % 4], label=label, zorder=l - i)

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if xticks:
        plt.xticks(xticks)
    if ypc:
        from matplotlib import ticker
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    if len(yKeys) and legend:
        plt.legend(loc="upper left", framealpha=1, frameon=False, **legend)

    plt.tight_layout()

    if filename:
        from ml_logger import logger
        logger.savefig(filename, dpi=300, transparent=True)

    plt.show()


@mlc
def plot_scatter(data_frame, xKey, *yKeys, title=None,
                 figsize=(3.8, 2.4),
                 legend=dict(bbox_to_anchor=(0.95, 0.8), fontsize=10),
                 size=None,
                 xlabel=None, ylabel=None, ynorm=False, ypc=False,
                 xlim=None, ylim=None,
                 filename=None, show=True):
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize, dpi=300)
    if title:
        plt.title(title, pad=15)
    if ynorm:
        data_frame = data_frame.copy()
        for i, yKey in enumerate(yKeys):
            ys = data_frame[yKey]
            data_frame[yKey] = ys / ys.rolling(20).mean().max()

    for i, yKey in enumerate(yKeys):
        s = data_frame[[xKey, yKey]].dropna()

        label = yKey.replace('_', " ")
        xs, ys = s[xKey], s[yKey]
        plt.scatter(xs, ys, color=COLORS[i % 4], alpha=0.7, s=size, edgecolor='none', label=label)

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if ypc:
        from matplotlib import ticker
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    if len(yKeys) and legend:
        plt.legend(loc="upper left", framealpha=1, frameon=False, **legend)
    plt.tight_layout()

    if filename:
        from ml_logger import logger
        logger.savefig(filename)

    plt.show()


@mlc
def plot_bernoulli(df, xKey, yKey, window=10, title=None, figsize=(3.4, 2.4), filename=None):
    """
    use window to get the binned average, and use p(1-p) to compute the variance.
    :param xs:
    :param ys:
    :return:
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    from scipy import interpolate

    xs = df[xKey]
    smoothed = df[yKey].rolling(window, min_periods=1).mean().interpolate(method='cubic')

    spline = interpolate.UnivariateSpline(xs, smoothed)
    spline.set_smoothing_factor(0.015)
    ys = spline(xs)
    variance = ys * (1 - ys) / window

    plt.figure(figsize=figsize, dpi=300)
    if title:
        plt.title(title)
    plt.plot(xs, ys, color="green")
    plt.fill_between(xs, ys - variance, ys + variance, color="green", alpha=0.5)
    plt.ylim(-0.05, 1.05)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()

    if filename:
        from ml_logger import logger
        logger.savefig(filename)
        logger.print('I saved a file!', filename)

    plt.show()
