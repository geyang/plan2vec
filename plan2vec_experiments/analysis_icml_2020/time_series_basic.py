import pandas as pd
import numpy as np


def plot_line(data_frame, xKey, *yKeys, title=None, raw=True, window=5, std=True):
    import matplotlib.pyplot as plt
    # plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    plt.figure(figsize=(3.8, 2.4), dpi=300)

    if title:
        plt.title(title)

    for i, yKey in enumerate(yKeys):
        s = data_frame[[xKey, yKey]].dropna()
        xs, ys = s[xKey], s[yKey]
        if raw:
            plt.plot(xs, ys, color='grey', alpha=0.7, linewidth=0.1, label=None if window else yKey)

        if window:
            smoothed = ys.rolling(window)
            plt.plot(xs, smoothed.mean(), color=colors[i % 4], label=yKey)

            p, m = ys + smoothed.std(), ys - smoothed.std()
            plt.fill_between(xs, y1=m, y2=p, alpha=0.3, color=colors[i % 4])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(loc="upper left", bbox_to_anchor=(0.95, 0.8), framealpha=1, frameon=False, fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = pd.DataFrame(dict(
        xValue=np.arange(200),
        yValue=0.12 * (np.arange(200) ** .3 + np.random.rand(200)),
        yValue_2=0.06 * (np.arange(200) ** .3 + np.random.rand(200)),
    ))

    colors = ['#49b8ff', '#ff7575', '#66c56c', '#f4b247']
    line_style = ['-', '-.', '--', ':']

    xKey = "xValue"
    yKey = "yValue"

    plot_line(df, xKey, "yValue", "yValue_2", title="Success Rate")
