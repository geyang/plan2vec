from plan2vec.plan2vec.plan2vec_img import Args, DEBUG, main
from params_proto.neo_hyper import Sweep
from plan2vec_experiments import instr, config_charts
import jaynes


def common_config():
    Args.start_seed = 5 * 100
    Args.lr = 3e-5

    Args.gamma = 0.97
    Args.target_update = 0.9
    Args.top_k = None
    Args.plan_steps = 1
    Args.H = 20
    Args.r_scale = 0.2
    Args.n_rollouts = 400
    Args.timesteps = 2

    Args.latent_dim = 3

    # turn off checkpointing b/c models are large
    Args.checkpoint_interval = None

    # Args.optim_batch_size = 128
    # Args.optim_epochs = 32
    Args.optim_epochs = 16
    Args.optim_batch_size = 32

    Args.num_epochs = 200  # we learn in 20 epochs..
    Args.visualization_interval = 1

    Args.env_id = "GoalMassDiscreteImgIdLess-v0"
    local_metric_exp_path = "geyang/plan2vec/2019/06-21/goal-mass-image/goal_mass_img_local_metric/22.39/21.462733"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


# noinspection PyShadowingNames
def plan2vec():
    with Sweep(Args) as sweep:
        common_config()

        # Args.global_metric = 'ResNet18L2'

        # Args.binary_reward = False
        # DEBUG.real_r_distance = False
        # DEBUG.supervised_value_fn = False
        # Args.term_r = 1.5
        Args.term_r, DEBUG.ground_truth_success = 0.04, True
        with sweep.product:
            Args.plan_steps = range(1, 8)
            # Args.seed = range(10)

    for deps in sweep:
        # DEBUG.ground_truth_neighbor_r = 0.04
        thunk = instr(main, deps, __postfix=f"plan2vec/{Args.env_id}/plan_steps-({Args.plan_steps})")
        config_charts()
        jaynes.run(thunk)


def sptm():
    with Sweep(Args) as sweep:
        common_config()

        # Args.global_metric = 'ResNet18L2'
        Args.global_metric = Args.load_local_metric
        Args.load_global_metric = Args.load_local_metric

        # Args.binary_reward = False
        # DEBUG.real_r_distance = False
        # DEBUG.supervised_value_fn = False
        # Args.term_r = 1.5
        Args.term_r, DEBUG.ground_truth_success = 0.04, True
        # with sweep.product:
        #     Args.seed = range(10)

    for deps in sweep:
        # DEBUG.ground_truth_neighbor_r = 0.04
        thunk = instr(main, deps, __postfix=f"plan2vec/{Args.env_id}/plan_steps-({Args.plan_steps})")
        config_charts()
        jaynes.run(thunk)


# # noinspection PyShadowingNames
# def plan2vec_gt_neighbor_binary(seed):
#     Args.seed = seed
#     Args.binary_reward = True
#     DEBUG.real_r_distance = False
#     DEBUG.supervised_value_fn = False
#     Args.term_r, DEBUG.ground_truth_success = 0.04, True
#     # DEBUG.ground_truth_neighbor_r = 0.04
#     _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG), __postfix=f"binary/{Args.global_metric}/lr-({Args.lr})")
#     config_charts()
#     jaynes.run(_)
#
#
# # note: non-binary version need to use the real-r-distance.
# # noinspection PyShadowingNames
# def plan2vec_gt_neighbor_real_r(seed):
#     Args.seed = seed
#     Args.binary_reward = False
#     DEBUG.real_r_distance = True
#     DEBUG.supervised_value_fn = False
#     Args.term_r, DEBUG.ground_truth_success = 0.04, True
#     # DEBUG.ground_truth_neighbor_r = 0.04
#     _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG), __postfix=f"dense/{Args.global_metric}/lr-({Args.lr})")
#     config_charts()
#     jaynes.run(_)


if __name__ == "__main__":
    jaynes.config()

    plan2vec()
    jaynes.listen()

exit()

from params_proto.neo_hyper import Sweep

from plan2vec_experiments import instr, config_charts
from plan2vec.plan2vec.plan2vec_state import DEBUG, Args, main
import jaynes


def common_config():
    Args.num_epochs = 100

    Args.lr = 3e-4
    Args.weight_decay = 0

    Args.gamma = 1
    Args.target_update = 0.9
    Args.top_k = None

    Args.neighbor_r = 1.2
    Args.term_r = 1.
    Args.plan_steps = 1
    Args.H = 20
    Args.r_scale = 0.2

    Args.visualization_interval = 10

    Args.optim_epochs = 16
    Args.optim_batch_size = 32
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
        print(deps)

    exit()
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
            Args.start_seed = range(3)

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
    jaynes.config()
    # jaynes.config("local")

    plan2vec()
    # plan2vec_random_policy()
    # sptm_baseline()

    jaynes.listen()
