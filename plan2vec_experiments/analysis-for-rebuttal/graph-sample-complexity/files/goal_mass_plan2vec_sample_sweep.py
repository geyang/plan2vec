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
    Args.plan_steps = 3
    Args.H = 20
    Args.r_scale = 0.2

    Args.global_metric = "GlobalMetricMlp"

    Args.visualization_interval = 10

    Args.optim_epochs = 1

    Args.env_id = 'GoalMassDiscreteIdLess-v0'
    local_metric_exp_path = "geyang/plan2vec/2019/06-21/goal-mass/goal_mass_local_metric/20.47/21.907891"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


def plan2vec():
    from params_proto.neo_hyper import Sweep
    from ml_logger import logger

    seeds = [s * 100 for s in range(5)]

    with Sweep(Args, DEBUG) as sweep:
        common_config()

        Args.num_epochs = 2000
        Args.binary_reward = False

        DEBUG.supervised_value_fn = False
        DEBUG.real_r_distance = False

        with sweep.product:

            with sweep.zip:
                Args.n_rollouts = [1, 3, 5, 10, 20, 30, 50,
                                   100, 200, 300, 400, 500, 600, ]
                Args.num_envs = [1, 1, 1, 1, 10, 10, 10,
                                 20, 20, 20, 20, 20, 20]

            Args.start_seed = seeds

    for deps in sweep:
        _ = instr(main, deps, __postfix=f"n_rollouts-({Args.n_rollouts})")
        config_charts()
        logger.log_text("""
        We use the MLP global metric function for better performance
        
        ```python
        Args.global_metric = 'MlpGlobalMetric'
        ```
        """)
        jaynes.run(_)


if __name__ == "__main__":
    jaynes.config("cpu")

    plan2vec()

    jaynes.listen()
