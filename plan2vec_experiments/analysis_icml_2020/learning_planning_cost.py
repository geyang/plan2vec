import pandas as pd
import numpy as np

if __name__ == '__main__':
    from plan2vec_experiments import RUN
    from plan2vec_experiments.analysis_icml_2020 import stylize, plot_line, plot_bar

    cost = np.concatenate([np.ones(30) * 378, np.ones(170) * 56])
    cost += cost ** 1.2 * np.random.rand(200)
    df = pd.DataFrame(dict(
        index=np.arange(200),
        queries=cost
    ))

    prefix = "geyang/plan2vec/2020/01-20/plan2vec/maze_plan2vec/alg-sweep/23.06"

    from ml_logger import logger

    logger.configure(RUN.server, prefix, register_experiment=False)
    paths = logger.glob("**/metrics.*")
    print(paths)

    exp_paths = {
        "SPTM/SoRB": 'dijkstra/46.717376/metrics.pkl',
        'UVPN': 'heuristic/53.637398/metrics.pkl',
        'UVPN*': 'a_star/53.210722/metrics.pkl',
    }

    cache = {}
    stack = []
    for k, v in exp_paths.items():
        data = pd.DataFrame(logger.load_pkl(v))
        series = data[['epoch', 'cost/mean']].dropna()
        stack.append(series.rename(columns={"cost/mean": k}))
        tail = series.sort_values('epoch')['cost/mean'].tail(100)
        cache[k] = tail.to_numpy().reshape(10, 10).mean(-1)

    import pandas as pd

    df = pd.concat(stack, axis=0, sort=False)

    stylize()
    plot_line(df, "epoch", 'UVPN', "UVPN*", 'SPTM/SoRB',  # title="Search Cost",
              figsize=(3.4, 2.2),
              xlabel="Epochs", ylabel="Node Expansion", ynorm=False,
              legend=dict(bbox_to_anchor=(0.028, 0.38),
                          ncol=1, columnspacing=1, labelspacing=0.25,
                          handlelength=2, fontsize=8),
              smooth=100,
              ylim=(0, None),
              filename="figures/cost_during_learning.png")

    keys = ["SPTM", 'UVPN', 'UVPN*']

    stylize()
    plot_bar(keys, cache.values(),  # title="Planning Cost",
             figsize=(1.7, 2.4),
             # xlabel=' ',
             xticks_rotation=40,
             filename="figures/cost_comparison_maze.png")
