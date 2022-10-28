def plot_facet(data_frame, xkey=None, ykey=None,
               groupby='key', groups=None,
               facet_key=None, facet=None, labels: dict = None,
               xscale=None, yscale=None,
               xlim=None, ylim=None,
               xlabel=None, ylabel=None,
               xtick=None, ytick=None,
               xmajor=None, ymajor=None,
               title=None,
               colors=('#49b8ff', '#ff7575', '#f4b247', '#66c56c',),
               line_styles=('-', '-.', '--', ':'),
               file_name=None, show=True):
    import seaborn as sns
    from matplotlib import rcParams
    import matplotlib.pyplot as plt

    rcParams['axes.titlepad'] = 15
    rcParams['axes.labelpad'] = 5

    plt.figure(figsize=(4.7, 2.5), dpi=100)

    try:
        grouped = data_frame.groupby(groupby)
    except KeyError:
        print(data_frame)
        return

    # todo: change these keys to be set automatically.
    logger.print(f"these keys are available: ", *[k for k, g in grouped], color="green")
    for i, key in enumerate(groups or grouped.groups.keys()):
        grp = grouped.get_group(key)
        try:
            # todo: support custom key, use `labels` parameter
            ax = sns.lineplot(x=xkey, y=ykey, data=grp, label=key, color=colors[i], )
            if xscale:
                plt.gca().set_xscale(xscale)
            if yscale:
                plt.gca().set_yscale(yscale)
            ax.lines[-1].set_marker('o')
            ax.lines[-1].set_linestyle(line_styles[i])
            if xtick:
                ax.xaxis.set_ticks(xtick)
            if xmajor:
                ax.xaxis.set_major_formatter(xmajor)
            if ytick:
                ax.yaxis.set_ticks(ytick)
            if ymajor:
                ax.yaxis.set_major_formatter(ymajor)
        except Exception as e:
            logger.print(f"key '{key}' errored out: {e}", color="red")

    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1), framealpha=1,
               frameon=False, fontsize=12)

    if title is not None:
        if facet is not None:
            plt.title(f"{title} [{facet}]", loc='left')
        else:
            plt.title(f"{title}", loc='left')

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if ylim:
        plt.ylim(*ylim)
    if xlim:
        plt.xlim(*xlim)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    if file_name:
        file_name = file_name.format(facet=facet, key=key, title=title)
        path = logger.savefig(file_name, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    logger.print(f'finished plotting. saved at {path}', color='green')


if __name__ == "__main__":

    from os.path import basename
    import pandas as pd
    from plan2vec_experiments import RUN, dir_prefix
    from ml_logger import logger

    logger.configure(RUN.server,
                     prefix=dir_prefix() + "/" + basename(__file__)[:-3])

    with logger.PrefixContext(".."):
        metrics_paths = logger.glob("**/metrics.pkl")
        # print(*metrics_paths, sep="\n")


    def load_experiment(metrics_path):
        from copy import copy
        local_logger = copy(logger)

        with local_logger.PrefixContext("..", metrics_path, "../"):

            if "plan2vec" in metrics_path:
                key = "plan2vec"
                # todo: change to start_seed later
                seed_key = "Args.seed"
                metric_key = "success_rate/mean"
            elif "dqn" in metrics_path:
                key = "dqn"
                seed_key = "Args.start_seed"
                metric_key = "eval/success_rate/mean"
            else:
                return

            try:
                n_rollouts, seed = local_logger.get_parameters(
                    "Args.n_rollouts", seed_key, path="parameters.pkl")

                metrics = pd.DataFrame(local_logger.load_pkl("metrics.pkl"))
                avg_success = metrics[metric_key][-100:].mean()

                local_logger.store_metrics(key=key, n_rollouts=n_rollouts, success=avg_success)

            except:
                pass

    load_experiment(metrics_paths[0])


    from multiprocessing.pool import ThreadPool
    from tqdm import tqdm

    pool = ThreadPool(20)
    pool.map(load_experiment, metrics_paths)

    import matplotlib.ticker as mtick

    # now plot the figures
    metrics_data = pd.DataFrame(logger.summary_cache.data)
    plot_facet(metrics_data, title="Success vs Sample Size",
               xkey="n_rollouts", ykey="success", groupby="key",
               groups=["plan2vec", 'dqn'],
               xlabel="# of Rollouts",
               ylabel="Success Rate",
               xscale="log",
               xlim=(10, 600),
               ylim=(-0.1, 1.1),
               xtick=[5, 10, 100, 1000],
               # xmajor=mtick.ScalarFormatter(),
               ymajor=mtick.PercentFormatter(1.0),
               file_name="figures/sample_complexity.png")
