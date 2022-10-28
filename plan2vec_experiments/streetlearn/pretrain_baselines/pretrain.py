from plan2vec_experiments import instr, config_charts
from plan2vec.plan2vec.plan2vec_streetlearn_2 import DEBUG, Args, main
import jaynes


def common_config():
    Args.seed = 5 * 100

    Args.num_epochs = 500
    Args.lr = 3e-5
    Args.gamma = 0.97
    Args.target_update = 0.9
    Args.top_k = None
    Args.plan_steps = 1
    Args.H = 50
    Args.r_scale = 0.2

    Args.optim_epochs = 32

    Args.latent_dim = 3

    # make this one to see early stage to make sure
    Args.visualization_interval = 10
    # turn off checkpointing b/c models are large
    Args.checkpoint_interval = None
    Args.binary_reward = None

    # DEBUG.pretrain_num_epochs = 1

    # DEBUG.pretrain_global = True
    # # also try to pretrain with random sample,
    # # try sampling techniques
    # # add evaluation loops
    # DEBUG.value_fn_pretrain_global = True

    # Args.data_path = None
    local_metric_exp_path = "episodeyang/plan2vec/2019/06-20/streetlearn/local_metric/23.19/07.247751"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric_400.pkl"


def plan2vec_coord_pretrain(dataset, prefix):
    Args.data_path = f"~/fair/streetlearn/processed-data/{dataset}"

    DEBUG.pretrain_global = True
    DEBUG.value_fn_pretrain_global = False
    DEBUG.supervised_value_fn = True
    DEBUG.value_fn_pretrain_goal_r = None

    assert Args.binary_reward is None
    assert DEBUG.oracle_planning is False
    Args.term_r, DEBUG.ground_truth_success = 2e-4, True
    DEBUG.ground_truth_neighbor_r = 2e-4
    DEBUG.real_r_distance = False
    _ = instr(main, __postfix=f"coord-pretrain/{prefix}", **vars(Args), _DEBUG=vars(DEBUG), __up=-1)
    config_charts(path="coord-pretrain.charts.yml")
    jaynes.run(_)


def plan2vec_value_pretrain(dataset, prefix):
    Args.data_path = f"~/fair/streetlearn/processed-data/{dataset}"

    DEBUG.pretrain_global = False
    DEBUG.value_fn_pretrain_global = True
    DEBUG.supervised_value_fn = True
    DEBUG.value_fn_pretrain_goal_r = None

    assert Args.binary_reward is None
    assert DEBUG.oracle_planning is False
    Args.term_r, DEBUG.ground_truth_success = 2e-4, True
    DEBUG.ground_truth_neighbor_r = 2e-4
    DEBUG.real_r_distance = False
    _ = instr(main, __postfix=f"value-pretrain/{prefix}", **vars(Args), _DEBUG=vars(DEBUG), __up=-1)
    config_charts(path="value-pretrain.charts.yml")
    jaynes.run(_)


def plan2vec_value_10_step_pretrain(dataset, prefix):
    Args.data_path = f"~/fair/streetlearn/processed-data/{dataset}"

    DEBUG.pretrain_global = False
    DEBUG.value_fn_pretrain_global = True
    DEBUG.value_fn_goal_distance = 10
    DEBUG.supervised_value_fn = True
    DEBUG.value_fn_pretrain_goal_r = 2e-4 * 6

    assert Args.sample_goal_r, "sample_goals_r need to be on."

    assert Args.binary_reward is None
    assert DEBUG.oracle_planning is False
    Args.term_r, DEBUG.ground_truth_success = 2e-4, True
    DEBUG.ground_truth_neighbor_r = 2e-4
    DEBUG.real_r_distance = False
    _ = instr(main, __postfix=f"value-10-step-pretrain/{prefix}", **vars(Args), _DEBUG=vars(DEBUG), __up=-1)
    config_charts(path="value-pretrain.charts.yml")
    jaynes.run(_)


def plan2vec_coord_value_pretrain(dataset, prefix):
    Args.data_path = f"~/fair/streetlearn/processed-data/{dataset}"

    DEBUG.pretrain_global = True
    DEBUG.value_fn_pretrain_global = True
    DEBUG.supervised_value_fn = True

    assert Args.binary_reward is None
    assert DEBUG.oracle_planning is False
    Args.term_r, DEBUG.ground_truth_success = 2e-4, True
    DEBUG.ground_truth_neighbor_r = 2e-4
    DEBUG.real_r_distance = False
    _ = instr(main, __postfix=f"coord-value-pretrain/{prefix}", **vars(Args), _DEBUG=vars(DEBUG), __up=-1)
    config_charts(path="coord-value-pretrain.charts.yml")
    jaynes.run(_)


if __name__ == "__main__":
    import numpy as np

    common_config()

    param_dict = {
        'ResNet18L2': {
            "lr": [1e-6, 3e-6, 1e-7, 3e-7],
        },
        # 'GlobalMetricConvL2_s1': {"lr": [1e-6, 3e-6, 6e-6]},
        # 'GlobalMetricConvDeepL2': {"lr": [1e-6, 3e-6, 6e-6]},
        # 'GlobalMetricConvDeepL2_wide': {"lr": [1e-6, 3e-6, 6e-6]}
    }

    # ResNet requires much less memory than the other.
    Args.global_metric = 'ResNet18L2'
    _ = param_dict['ResNet18L2']

    jaynes.config("vector-gpu")

    for key in ['tiny', 'small', 'medium', 'large', 'xl']:
        for lr in _['lr']:
            DEBUG.pretrain_lr = lr
            DEBUG.value_fn_pretrain_lr = lr
            Args.lr = lr / 10.

            plan2vec_value_pretrain(f"manhattan-{key}",
                                    f"manhattan-{key}/{Args.global_metric}/lr-({lr})")
            plan2vec_coord_pretrain(f"manhattan-{key}",
                                    f"manhattan-{key}/{Args.global_metric}/lr-({lr})")
            plan2vec_value_pretrain(f"manhattan-{key}",
                                    f"manhattan-{key}/{Args.global_metric}/lr-({lr})")
            plan2vec_coord_value_pretrain(f"manhattan-{key}",
                                          f"manhattan-{key}/{Args.global_metric}/lr-({lr})")

    jaynes.listen()
