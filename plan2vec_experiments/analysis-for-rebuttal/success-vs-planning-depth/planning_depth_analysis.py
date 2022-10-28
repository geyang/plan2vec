import pandas as pd
from plan2vec_experiments import instr, RUN, dir_prefix, config_charts
from ml_logger import logger


def list_all():
    logger.configure(log_directory=RUN.server, prefix=dir_prefix() + "/analysis")

    root_dir = dir_prefix().replace('01-08', '01-04')

    with logger.PrefixContext(root_dir):
        metrics = logger.glob("**/metrics.pkl")

    return root_dir, metrics


def clean_cache():
    import time
    logger.configure(log_directory=RUN.server, prefix=dir_prefix() + "/analysis",
                     register_experiment=False)
    logger.remove(".cache/planning_data.pkl")

    time.sleep(1)


def get_data(root_dir, metrics):
    logger.configure(log_directory=RUN.server, prefix=dir_prefix() + "/analysis",
                     register_experiment=False)

    parameter_keys = 'Args.env_id', 'Args.plan_steps',

    for metrics_path in metrics:
        with logger.PrefixContext(root_dir):
            exp_path = '/'.join(metrics_path.split('/')[:-1])

            if "random_policy" in exp_path:
                key = "random"
            elif "MlpGlobalMetric" in exp_path:
                key = "plan2vec"
            elif "sptm" in exp_path:
                key = "sptm"
            else:
                continue

            env_id, plan_steps = logger.get_parameters(*parameter_keys, path=exp_path + "/parameters.pkl")
            df = pd.DataFrame(logger.load_pkl(metrics_path))

        try:
            # last_success = df['success_rate/mean'][-100:]
            # avg_success = last_success.mean()

            success = df['success_rate/mean'].rolling(100).mean()
            import numpy as np
            success = np.nanmax(success.values)

            logger.store_metrics(key=key, env_id=env_id, plan_steps=plan_steps, success=success,
                                 prefix=exp_path)

        except FileNotFoundError:
            print(f"{exp_path}/parameters.pkl not exis")
        except KeyError:
            print(f"{exp_path} does not work")

    # logger.peek_stored_metrics(len=None)

    df = pd.DataFrame(logger.summary_cache.data)
    logger.log_data(df, ".cache/planning_data.pkl")


def visualize():
    from functools import reduce
    logger.configure(log_directory=RUN.server, prefix=dir_prefix() + "/analysis")

    dfs = logger.load_pkl(".cache/planning_data.pkl")

    all_df = reduce(lambda left, right: pd.merge(left, right, how='outer'), dfs)

    for_env = all_df[all_df['env_id'] == "GoalMassDiscreteImgIdLess-v0"]

    # for_env = all_df[all_df['env_id'] == "GoalMassDiscreteIdLess-v0"]
    # logger.log(f"number of experiments: {len(for_env)}", color="green")
    # len(for_env[for_env['key'] == "plan2vec"])
    # for_env = all_df[all_df['env_id'] == "CMazeDiscreteIdLess-v0"]
    # len(for_env[for_env['key'] == "plan2vec"])
    # print(for_env[for_env['key'] == "plan2vec"].to_csv())
    #
    # logger.log(f"number of experiments: {len(for_env)}", color="green")
    # for_env = all_df[all_df['env_id'] == "回MazeDiscreteIdLess-v0"]
    # logger.log(f"number of experiments: {len(for_env)}", color="green")
    # logger.flush()
    #
    # exit()

    # plot_facet(all_df[all_df['env_id'] == "GoalMassDiscreteIdLess-v0"])
    # for_env = all_df[all_df['env_id'] == "CMazeDiscreteIdLess-v0"]
    # print(len(for_env))
    # plot_facet(for_env, env="Wall")
    for_env = all_df[all_df['env_id'] == "GoalMassDiscreteIdLess-v0"]
    print(len(for_env))
    plot_facet(for_env, env="Open Room")
    for_env = all_df[all_df['env_id'] == "CMazeDiscreteIdLess-v0"]
    print(len(for_env))
    plot_facet(for_env, env="Wall")


def plot_facet(data_frame, env=None):
    import seaborn as sns
    from matplotlib import rcParams
    import matplotlib.ticker as mtick
    import matplotlib.pyplot as plt

    rcParams['axes.titlepad'] = 15
    rcParams['axes.labelpad'] = 5

    fig = plt.figure(figsize=(4.7, 2.5), dpi=100)
    colors = ['#49b8ff', '#f4b247', '#ff7575', '#66c56c', ]
    line_styles = ['-', '-.', '--', ':']
    grouped = data_frame.groupby('key')
    for i, key in enumerate(["plan2vec", "SPTM", "random"]):
        try:
            grp = grouped.get_group(key.lower())
            ax = sns.lineplot(x="plan_steps", y="success", data=grp, label=key, color=colors[i], )
            ax.lines[-1].set_marker('o')
            ax.lines[-1].set_linestyle(line_styles[i])
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        except:
            logger.print(f"key '{key}' is missing from the dataset", color="yellow")
            pass

    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1), framealpha=1, frameon=False, fontsize=12)

    plt.title(f"Success vs Planning Steps [{env}]", loc='left')
    plt.xlabel("Planning Steps")
    plt.ylabel('Success Rate')
    plt.ylim(0, 1.05)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    path = logger.savefig(f'figures/success_vs_planning_{env.lower()}.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    logger.print(f'finished plotting. saved at {path}', color='green')


if __name__ == '__main__':
    # single threaded data pulling
    # root_dir, metrics = list_all()
    # clean_cache()
    # get_data(root_dir, metrics)

    # from functools import partial
    # from more_itertools import chunked
    # from multiprocessing import Pool
    #
    # pool = Pool(10)
    #
    # root_dir, metrics = list_all()
    # clean_cache()
    # pool.map(partial(get_data, root_dir), chunked(metrics, 5))
    # logger.print('finished all data processing', color="green")

    visualize()
