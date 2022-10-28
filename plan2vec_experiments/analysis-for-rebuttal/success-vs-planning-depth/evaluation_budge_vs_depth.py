from plan2vec_experiments import instr, config_charts
from plan2vec.plan2vec.plan2vec_state import DEBUG, Args, main
import jaynes


def common_config():
    Args.num_epochs = 5000

    Args.lr = 3e-4
    Args.weight_decay = 0

    Args.gamma = 1
    Args.target_update = 0.9
    Args.top_k = None

    Args.neighbor_r = 1.
    Args.term_r = 1.
    Args.plan_steps = 1
    Args.H = 20
    Args.r_scale = 0.2

    Args.global_metric = "GlobalMetricMlp"

    Args.visualization_interval = 10

    Args.optim_epochs = 32
    # Args.optim_batch_size = 128
    # Args.optim_epochs = 50
    # Args.batch_n = 100

    Args.env_id = 'GoalMassDiscreteIdLess-v0'
    local_metric_exp_path = "geyang/plan2vec/2019/06-21/goal-mass/goal_mass_local_metric/20.47/21.907891"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


def log_neighborhood_size():
    common_config()

    Args.binary_reward = False
    DEBUG.supervised_value_fn = False
    DEBUG.real_r_distance = False

    DEBUG.collect_neighborhood_size = True
    # exit right after the initial logging.
    Args.num_epochs = 0

    for Args.plan_steps in [1, 2, 3, 4, 5, 6, 7]:
        _ = instr(main, __postfix=f"neighborhood-size-vs-plan-depth/plan_steps-({Args.plan_steps})",
                  **vars(Args), _DEBUG=vars(DEBUG))
        config_charts()
        jaynes.run(_)


def analysis():
    import jaynes
    from plan2vec_experiments import RUN, dir_prefix
    from ml_logger import logger

    jaynes.config('vector')

    parameter_keys = 'Args.plan_steps',

    logger.configure(log_directory=RUN.server, prefix=dir_prefix() + "/planning_cost_analysis")

    with logger.PrefixContext(dir_prefix()):
        data_files = logger.glob("**/neighborhood_size.pkl")
        print(*data_files, sep="\n")

    logger.print(*parameter_keys, 'neighborhood_size', sep=',\t', file='results.csv')

    for path in data_files:
        with logger.PrefixContext(dir_prefix()):
            exp_path = '/'.join(path.split('/')[:-1])
            parameter_values = logger.get_parameters(*parameter_keys, path=exp_path + "/parameters.pkl")
            data, = logger.load_pkl(path)

            import numpy as np
            data = np.array(data)

        logger.print(parameter_values, data.mean(), sep=',\t', file='results.csv')


if __name__ == "__main__":
    # jaynes.config("vector")

    # log_neighborhood_size()
    # jaynes.listen()

    analysis()
