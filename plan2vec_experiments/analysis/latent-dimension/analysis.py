import pandas as pd
from params_proto import cli_parse

from plan2vec_experiments import instr


@cli_parse
class Config:
    sweep_root = "geyang/plan2vec/2019/07-10/analysis/latent-dimention/" \
                 "latent_dim_long_term/binary_reward"


def single_run_stats(exp_path, x_key, y_key, x_bin_size=100):
    import numpy as np
    from ml_logger import logger

    metrics = pd.DataFrame(logger.load_pkl(exp_path + '/metrics.pkl'))

    s = metrics[[y_key, x_key]].set_index(x_key)
    ticks = np.arange(0, len(s), x_bin_size)
    bins = pd.cut(s.index, ticks)
    return s.groupby(bins)[y_key].agg({
        'mean': lambda c: c.mean(),
        "25%": lambda c: c.quantile(.25),
        "75%": lambda c: c.quantile(.75)
    }).set_index(ticks[1:])

    # s = _['success_rate/mean']
    # import matplotlib.pyplot as plt
    # plt.plot(s.index, s['mean'], color=color, label=label)
    # plt.fill_between(s.index, y1=s["25%"], y2=s["75%"], alpha=0.3, color=color)
    # plt.ylim(-0.1, 1.1)
    # plt.xlim(-10, 1010)


colors = ['#49b8ff', '#ff7575', '#66c56c', '#f4b247']
line_style = ['-', '-.', '--', ':']


def collect_data():
    import os
    from ml_logger import logger, ML_Logger

    parameters = "Args.env_id", "Args.global_metric", "Args.latent_dim"
    rows = []
    with logger.PrefixContext(Config.sweep_root):
        run_paths = logger.glob("**/parameters.pkl")
        print(*run_paths, sep='\n')

        for i, p in enumerate(run_paths):
            exp_path = os.path.dirname(p)

            try:
                params = logger.get_parameters(*parameters, path=exp_path + "/parameters.pkl")
            except:
                continue

            _ = single_run_stats(exp_path, "epoch", "success_rate/mean")

            idx = _['mean'].idxmax()
            success_rate = f"{_['mean'][idx] * 100:0.1f}"
            spread = f"{(_['75%'][idx] - _['25%'][idx]) * 50:0.1f}"

            rows.append([*params, success_rate, spread])

    df = pd.DataFrame(rows, columns=[*parameters, "success_rate (%)", "±25%"])
    logger.log_text(df.to_csv(index=False), 'results.csv', overwrite=True)


def make_bar_chart():
    import io
    import matplotlib.pyplot as plt
    import pandas as pd

    _ = logger.glob('results.csv')

    text = logger.load_text("results.csv")
    df = pd.read_csv(io.StringIO(text))
    pivot = df.pivot(index='Args.latent_dim', columns='Args.env_id')
    pivot['success_rate (%)'].plot \
        .bar(yerr=pivot['±25%'] * 2, width=0.9, subplots=True, ylim=[0, 100], legend=False,
             color=colors)
    logger.savefig(f"figures/success_rate_vs_env_id.png")
    logger.savefig(f"figures/success_rate_vs_env_id.pdf")


if __name__ == "__main__":
    from ml_logger import logger
    from plan2vec_experiments import RUN

    logger.configure(RUN.server, prefix=f"{Config.sweep_root}/analysis")
    logger.upload_file(__file__)
    collect_data()
    make_bar_chart()
