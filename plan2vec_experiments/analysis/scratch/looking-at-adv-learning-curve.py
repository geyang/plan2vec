from ml_logger import logger
from params_proto.neo_proto import ParamsProto


class Analysis(ParamsProto):
    exp_root = "/geyang/plan2vec/2020/01-24/neo_plan2vec/tweaking_maze_adv/maze_env_sweep"


def read_data(index, exp_prefix):
    from ml_logger import logger

    with logger.PrefixContext(Analysis.exp_root, exp_prefix, ".."):
        metrics = logger.load_pkl("metrics.pkl")
        env_id = logger.get_parameters("Args.env_id", path="parameters.pkl")

    if metrics is None:
        logger.print(f"{index}: {exp_prefix} is empty", color="red")

    return env_id, metrics


def plot(env_id, metrics, index=None, path=None):
    logger.print(f"Data length: {len(metrics)}", color="green")

    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    df = pd.DataFrame(metrics)

    df.keys()

    plt.figure(figsize=(4, 3), dpi=300)
    plt.title('Success Rate')
    _ = df[['epoch', 'success/mean']].dropna()
    logger.print(f"Data length: {len(_)}", color="green")
    ax = plt.plot(_['epoch'], _['success/mean'], color="#23aaff", alpha=0.6, linewidth=3)
    plt.xlabel('Epoch')
    plt.ylabel('Success Rate')
    ax[-1].set_marker('o')
    # ax.lines[-1].set_linestyle(line_styles[i])
    # ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    logger.savefig(f'./figures/success_rate_{env_id}_{index:02d}.png')
    # plt.show()
    plt.close()

    if path:
        logger.print(f"{index:02d}_{path}")
    logger.print(f'Maximum Success Rate is {_["success/mean"].max():.2%}', color="green")

    print("done")

    # Results:
    # Data length: 1200
    # Data length: 119
    # Maximum Success Rate is 90.00%
    # done


def run():
    from multiprocessing.pool import ThreadPool
    from plan2vec_experiments import RUN, dir_prefix

    logger.log_params(Args=vars(Analysis))

    with logger.PrefixContext(Analysis.exp_root):
        exp_paths = logger.glob("**/metrics.pkl")

    print(*exp_paths, sep="\n")

    pool = ThreadPool(1)
    results = pool.starmap(read_data, enumerate(exp_paths))

    for index, [(env_id, metrics), path] in enumerate(zip(results, exp_paths)):
        if metrics is not None:
            plot(env_id, metrics, index, path)


if __name__ == '__main__':
    from plan2vec_experiments import instr, config_charts

    # Analysis.exp_root = "/geyang/plan2vec/2020/01-25/neo_plan2vec/tweaking_maze_adv/maze_env_sweep/17.31"
    # Analysis.exp_root = "/geyang/plan2vec/2020/01-25/neo_plan2vec/tweaking_maze_adv/maze_env_sweep/18.57"
    # Analysis.exp_root = "/geyang/plan2vec/2020/01-26/neo_plan2vec/tweaking_maze_adv/maze_env_sweep/backpressure/11.18"
    # Analysis.exp_root = "/amyzhang/plan2vec/2020/01-26/neo_plan2vec/tweaking_maze_adv/maze_env_sweep/backpressure/12.13"
    # Analysis.exp_root = "/amyzhang/plan2vec/2020/01-26/neo_plan2vec/tweaking_maze_adv/maze_load_adv_tweak/bp-tweak/16.05"
    # Analysis.exp_root = "/amyzhang/plan2vec/2020/01-26/neo_plan2vec/tweaking_maze_adv/maze_load_adv_tweak/bp-tweak-avg-n-target/16.36"
    # Analysis.exp_root = "/amyzhang/plan2vec/2020/01-26/neo_plan2vec/tweaking_maze_adv/maze_load_adv_tweak/bp-tweak-avg-neighbor/16.21"
    # Analysis.exp_root = "/amyzhang/plan2vec/2020/01-26/neo_plan2vec/tweaking_maze_adv/maze_load_adv_tweak/bp-tweak-flex-mean/19.24"
    # Analysis.exp_root = "/amyzhang/plan2vec/2020/01-27/neo_plan2vec/tweaking_maze_adv/maze_load_adv_tweak/longer"
    # Analysis.exp_root = "/amyzhang/plan2vec/2020/01-28/neo_plan2vec/tweaking_maze_adv/maze_load_adv_tweak/longer/2xv_s_s/07.44/GoalMassDiscreteImgIdLess-v0/ams-0.1"
    Analysis.exp_root = "/amyzhang/plan2vec/2020/01-28/neo_plan2vec/tweaking_maze_adv/maze_load_adv_tweak/longer/2xv_s_s/07.44/CMazeDiscreteImgIdLess-v0/ams-0.1"

    thunk = instr(run)
    config_charts("""
        charts:
        - type: image
          glob: "**/success_rate*.png"
        keys:
        - Args.exp_root
        """)
    thunk()
