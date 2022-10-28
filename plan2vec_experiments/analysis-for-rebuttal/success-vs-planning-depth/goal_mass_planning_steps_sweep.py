from params_proto.neo_hyper import Sweep

from plan2vec_experiments import instr, config_charts
from plan2vec.plan2vec.plan2vec_state import DEBUG, Args, main
import jaynes


def common_config():
    Args.num_epochs = 5000

    Args.lr = 3e-4
    Args.weight_decay = 0

    Args.num_epochs = 100

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


def plan2vec():
    with Sweep(Args, DEBUG) as sweep:
        common_config()

        with sweep.product:
            Args.plan_steps = [1, 2, 3, 4, 5, 6, 7]
            Args.start_seed = range(3)

    for deps in sweep:
        thunk = instr(main, deps, __postfix=f"MlpGlobalMetric-plan-depth-sweep/plan_steps-({Args.plan_steps})")
        config_charts()
        jaynes.run(thunk)


def plan2vec_random_policy():
    with Sweep(Args, DEBUG) as sweep:
        common_config()

        DEBUG.random_policy = True

        with sweep.product:
            Args.plan_steps = [1, 2, 3, 4, 5, 6, 7]
            Args.start_seed = range(10)

    for deps in sweep:
        thunk = instr(main, deps, __postfix=f"random_policy-plan-depth-sweep/plan_steps-({Args.plan_steps})")
        config_charts(path="oracle_planner.charts.yml")
        jaynes.run(thunk)


def sptm_baseline():
    from ml_logger import logger
    with Sweep(Args, DEBUG) as sweep:
        common_config()

        Args.global_metric = 'LocalMetric'
        Args.latent_dim = 50  # to be identical to the local metric function
        # this needs to be fixed
        local_metric_exp_path = "geyang/plan2vec/2019/06-21/goal-mass/goal_mass_local_metric/20.47/21.907891"
        Args.load_global_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"
        Args.lr = 0

        with sweep.product:
            Args.plan_steps = [1, 2, 3, 4, 5, 6, 7]
            Args.start_seed = range(10)

    for deps in sweep:
        thunk = instr(main, deps, __postfix=f"sptm-plan-depth-sweep/plan_steps-({Args.plan_steps})")
        config_charts(path="oracle_planner.charts.yml")
        logger.log_text("""
        # SPTM baseline:
        
        The settings are:
        
        - use same global model as local
        - set the latent_dim to 50 so that global metric is identical to local metric
        - load weights from local metric
        - freeze everything else
    
        ```python
        Args.global_metric = 'LocalMetric'
        Args.latent_dim = 50  # to be identical to the local metric function
        Args.load_global_metric = Args.load_local_metric
        Args.lr = 0
        ```
        """, filename="README.md", dedent=True)
        jaynes.run(thunk)


if __name__ == "__main__":
    jaynes.config("vector")
    # jaynes.config("local")

    # plan2vec()
    # sptm_baseline()
    plan2vec_random_policy()

    jaynes.listen()
