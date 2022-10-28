import pandas as pd
from params_proto import cli_parse
from tqdm import tqdm
from typing import Sequence


@cli_parse
class Config:
    sweep_root = "geyang/plan2vec/2019/07-31/analysis/baselines/plan2vec_streetlearn_evaluation"


def single_run_stats(exp_path, x_key, y_key, x_bin_size=100):
    import numpy as np
    from ml_logger import logger

    metrics = pd.DataFrame(logger.load_pkl(exp_path + '/metrics.pkl'))
    if metrics.empty:
        logger.print(exp_path, " metric file is empty.", color="yellow")
        return None

    try:
        s = metrics[[y_key, x_key]].set_index(x_key)
    except KeyError:
        return None
    ticks = np.arange(0, len(s), x_bin_size)
    bins = pd.cut(s.index, ticks)
    return s.groupby(bins)[y_key].agg([
        ['mean', lambda c: c.mean()],
        ["25%", lambda c: c.quantile(.25)],
        ["75%", lambda c: c.quantile(.75)]
    ]).set_index(ticks[1:])

    # s = _['success_rate/mean']
    # import matplotlib.pyplot as plt
    # plt.plot(s.index, s['mean'], color=color, label=label)
    # plt.fill_between(s.index, y1=s["25%"], y2=s["75%"], alpha=0.3, color=color)
    # plt.ylim(-0.1, 1.1)
    # plt.xlim(-10, 1010)


colors = ['#49b8ff', '#ff7575', '#66c56c', '#f4b247']
line_style = ['-', '-.', '--', ':']


def get_params(path):
    from ml_logger import logger
    return logger.get_parameters(path=path, silent=True)


def collect_parameters(root, exp_paths):
    from os.path import join

    parameters = tqdm(map(get_params, [join("/", root, path) for path in exp_paths]))
    return pd.DataFrame(parameters)


# predicate = lambda df: (df['Args.binary_reward'] == True) \
#                        & (df['DEBUG.real_r_distance'] == False) \
#                        & (df['DEBUG.supervised_value_fn'] == False)

def collect_data(no_cache=False, include=None, exclude=("run*", "args", "kwargs*", "host*"), predicate=None,
                 groupby=None):
    from ml_logger import logger

    with logger.PrefixContext(Config.sweep_root):
        try:
            assert not no_cache, "re-run glob"
            run_paths = logger.load_text("parameters.glob").strip().split('\n')
            logger.print(f"found {len(run_paths)} experiments", color='green')
            assert run_paths is not None, "run_paths is None"
        except:
            run_paths = logger.glob("**/parameters.pkl")
            logger.log_line(*run_paths, sep="\n", file="parameters.glob", flush=True)

        try:
            assert not no_cache, "re-load parameters"
            params, = logger.load_pkl(key="sweep_parameters.pkl")
            logger.log_line(f"found {len(params)} parameters", color="green")
        except:
            params = collect_parameters(Config.sweep_root, run_paths).set_index("run.prefix")
            logger.log_data(params, "sweep_parameters.pkl", overwrite=True)

        def diff_params(params, exclude=None, include=None):
            import fnmatch

            keys = params.keys()
            filtered_keys = set(keys)
            if exclude:
                for e in [exclude] if isinstance(exclude, str) else exclude:
                    filtered_keys = filtered_keys.difference(set(fnmatch.filter(keys, e)))
            if include:
                for i in [include] if isinstance(include, str) else include:
                    filtered_keys = filtered_keys.union(set(fnmatch.filter(keys, i)))

            diff_params = {}
            for key in params[[*filtered_keys]]:
                try:
                    values = params[key]
                    uniques = set(values)
                except:
                    values = values.transform(lambda _: tuple(_) if isinstance(_, Sequence) else _)
                    uniques = set(values)

                counts = len(uniques)
                if 1 < counts < len(params[key]):
                    # uniques dedupes NaN's
                    diff_params[key] = values.unique()

            return dict(sorted(diff_params.items(), key=lambda kv: len(kv[1])))

        # filter
        df = params
        params.reset_index(inplace=True)
        if predicate:
            df = df.loc[predicate(df), :]
        diff_keys = diff_params(df, include=include, exclude=exclude)
        logger.pprint(diff_keys, sep="\n")

        keys = ["Args.env_id", *diff_keys.keys()]

        # df.reset_index(inplace=True)
        df.set_index('run.prefix', inplace=True)
        for run_prefix, params in tqdm(df.iterrows(), total=len(df.index), desc="loading parameter"):
            _ = single_run_stats("/" + run_prefix, "epoch", "success_rate", x_bin_size=20)
            if _ is None:
                continue
            df.at[run_prefix, 'success rate (%)'] = f"{_['mean'].mean() * 100:0.1f}"
            df.at[run_prefix, '±25%'] = f"{(_['75%'].mean() - _['25%'].mean()) * 50:0.1f}"

        try:
            df.dropna(subset=['success rate (%)'], inplace=True)
        except KeyError:
            return logger.print('no success rate is found in this folder.', color="red")

        def find_max(row):
            idx = row['success rate (%)'].astype(float).idxmax()
            return row.loc[idx]

        def mean(row):
            # todo: test this.
            return row.mean()

        # df.reset_index(inplace=True)
        for k in keys:
            df[k] = df[k].transform(lambda _: str(_) if isinstance(_, Sequence) else _).astype('category')

        df.reset_index(inplace=True)
        # df.set_index([k for k in diff_keys.keys() if k is not "Args.seed"], inplace=True)
        # todo: support list of {groupby, lambda, keys}
        # todo: metrics processing needs support.
        logger.log_data(df, "full_results.pkl")

        if groupby:
            try:
                result = df.groupby(groupby, as_index=False) \
                    .apply(find_max)[[*groupby, 'success rate (%)', '±25%', 'run.prefix']]
            except:
                print(df.keys())
                result = df[[*groupby, 'success rate (%)', '±25%', 'run.prefix']]
        else:
            result = df

    logger.print(result, flush=True)
    logger.log_text(result.to_csv(), 'results.csv', overwrite=True)


def make_bar_chart():
    import io
    import matplotlib.pyplot as plt
    import pandas as pd
    from ml_logger import logger

    text = logger.load_text("results.csv")
    df = pd.read_csv(io.StringIO(text))
    if df.empty:
        return logger.print("the csv file is empty.", color="yellow")

    pivot = df.pivot(index='Args.global_metric', columns='Args.env_id')
    pivot['success rate (%)'].plot \
        .bar(yerr=pivot['±25%'] * 2, width=0.9, subplots=True, ylim=[0, 100], legend=False,
             color=colors)
    logger.savefig(f"figures/success_rate_vs_env_id.png")
    logger.savefig(f"figures/success_rate_vs_env_id.pdf")
    plt.show()


def launch(sweep_root, **kwargs):
    import os
    from ml_logger import logger
    print(RUN.server)
    logger.configure(RUN.server, prefix=os.path.join(sweep_root, "analysis"))
    # logger.upload_file(__file__)

    Config.sweep_root = sweep_root
    collect_data(**kwargs)
    # make_bar_chart()


if __name__ == "__main__":
    import os
    from ml_logger import logger
    from plan2vec_experiments import RUN

    sweep_roots = [
        # "geyang/plan2vec/2019/07-31/analysis/baselines/plan2vec_streetlearn_evaluation",
        # "geyang/plan2vec/2019/07-31/analysis/baselines/random_streetlearn_evaluation"
        # "geyang/plan2vec/2019/07-31/analysis/baselines/plan2vec_img_evaluation"
        # "geyang/plan2vec/2019/07-31/analysis/baselines/sptm_greedy_evaluation"
        # "geyang/plan2vec/2019/07-31/baselines/sptm_greedy_maze"
        # "geyang/plan2vec/2019/07-31/baselines/vae_baselines/vae_greedy_maze"
        "geyang/plan2vec/2019/07-31/baselines/vae_baselines/vae_greedy_streetlearn"
    ]

    import jaynes

    # jaynes.config('vector', runner=dict(n_cpu=4))
    jaynes.config('local')
    for root in sweep_roots:
        logger.configure(RUN.server, prefix=os.path.join(root, "analysis"))
        logger.upload_file(__file__)

        jaynes.run(launch, sweep_root=root,
                   predicate=lambda df: ('run.status' in df) and df['run.status'] == 'completed',
                   groupby=['Args.env_id', ])

    jaynes.listen()
