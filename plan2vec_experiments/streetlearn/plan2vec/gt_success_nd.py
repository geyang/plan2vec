from plan2vec_experiments import instr, config_charts
from plan2vec.plan2vec.plan2vec_streetlearn_2 import DEBUG, Args, main
import jaynes


def common_config():
    Args.seed = 5 * 100

    Args.num_epochs = 500  # about 1.76 days
    # Args.lr = 3e-5
    Args.gamma = 0.97
    Args.target_update = 0.9
    Args.top_k = None
    Args.plan_steps = 1
    Args.H = 20
    Args.r_scale = 0.2

    Args.optim_epochs = 32
    
    Args.latent_dim = 8


    # make this one to see early stage to make sure
    Args.visualization_interval = 1
    # turn off checkpointing b/c models are large
    Args.checkpoint_interval = None

    # Args.load_global_metric = "/geyang/plan2vec/2019/07-15/streetlearn/pretrain_baselines/" \
    #                           "sample-distribution-comparison/value-05-step-pretrain/dim-(2)/" \
    #                           "manhattan-medium/ResNet18L2/lr-(7.5e-08)/22.16/31.668959/" \
    #                           "value_fn_pretrain/global_metric_fn.pkl"

    # Args.data_path = None
    local_metric_exp_path = "geyang/plan2vec/2019/07-31/streetlearn/local_metric/manhattan-xl/ResNet18L2/lr-(1e-05)/00.37/42.222274"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric_400.pkl"


def plan2vec_gt_neighbor(dataset, neighbor_r, success_r, prefix):
    Args.data_path = f"~/fair/streetlearn/processed-data/{dataset}"

    Args.binary_reward = None
    Args.plan_steps = 1
    DEBUG.ground_truth_neighbor_r = neighbor_r

    assert Args.load_global_metric is None
    assert DEBUG.pretrain_global is False
    assert DEBUG.supervised_value_fn is False
    assert DEBUG.value_fn_pretrain_global is False
    assert Args.binary_reward is None
    assert DEBUG.oracle_planning is False
    assert DEBUG.real_r_distance is False

    Args.term_r, DEBUG.ground_truth_success = success_r, True

    _ = instr(main, __postfix=f"gt_neighbor/{prefix}", **vars(Args), _DEBUG=vars(DEBUG))
    config_charts()
    jaynes.run(_)


def plan2vec_binary(dataset, steps, neighbor_r, success_r, prefix):
    Args.data_path = f"~/fair/streetlearn/processed-data/{dataset}"

    Args.binary_reward = True
    Args.plan_steps = steps
    DEBUG.ground_truth_neighbor_r = None

    assert Args.load_global_metric is None
    assert DEBUG.pretrain_global is False
    assert DEBUG.supervised_value_fn is False
    assert DEBUG.value_fn_pretrain_global is False
    assert DEBUG.oracle_planning is False
    assert DEBUG.real_r_distance is False

    Args.neighbor_r = neighbor_r
    Args.term_r, DEBUG.ground_truth_success = success_r, True

    DEBUG.real_r_distance = False
    _ = instr(main, __postfix=f"binary/{prefix}", **vars(Args), _DEBUG=vars(DEBUG))
    config_charts()
    jaynes.run(_)


def plan2vec(dataset, steps, neighbor_r, success_r, prefix):
    Args.data_path = f"~/fair/streetlearn/processed-data/{dataset}"

    Args.binary_reward = False
    Args.plan_steps = steps
    DEBUG.ground_truth_neighbor_r = None

    assert Args.load_global_metric is None
    assert DEBUG.pretrain_global is False
    assert DEBUG.supervised_value_fn is False
    assert DEBUG.value_fn_pretrain_global is False
    assert DEBUG.oracle_planning is False
    assert DEBUG.real_r_distance is False

    Args.neighbor_r = neighbor_r
    Args.term_r, DEBUG.ground_truth_success = success_r, True

    _ = instr(main, __postfix=f"non_binary/{prefix}", **vars(Args), _DEBUG=vars(DEBUG))
    config_charts()
    jaynes.run(_)


if __name__ == "__main__":

    param_dict = {
        'ResNet18L2': {"lr": [1e-6, 3e-6, 1e-7, 3e-7], },
        # 'GlobalMetricConvL2_s1': {"lr": [1e-6, 3e-6, 6e-6]},
        # 'GlobalMetricConvDeepL2': {"lr": [1e-6, 3e-6, 6e-6]},
        # 'GlobalMetricConvDeepL2_wide': {"lr": [1e-6, 3e-6, 6e-6]}
    }

    # note: we generate these by scaling the overall map to 1 x 1.
    r_scale_dict = {
        "tiny": 0.2,
        "small": 0.09,
        "medium": 0.06,
        "large": 0.04,
        "xl": 0.02,
    }

    # ResNet requires much less memory than the other.
    Args.global_metric = 'ResNet18L2'

    _ = param_dict['ResNet18L2']

    jaynes.config("vector-gpu")

    common_config()

    for key in ['tiny', 'small', 'medium', 'large']:
        for Args.lr in [1e-5, 3e-5, 1e-4]:
            r = 2
            n = 2
            plan2vec_gt_neighbor(f"manhattan-{key}", neighbor_r=1.2e-4 * float(n),
                                 success_r=1.2e-4 * float(r),
                                 prefix=f"dim-({Args.latent_dim})/manhattan-{key}/n-({n})/r-({r})/lr-({Args.lr})")

            neighbor_r = 0.9
            n = 3
            plan2vec_binary(f"manhattan-{key}", steps=n, success_r=1.2e-4 * float(r),
                            neighbor_r=neighbor_r,
                            prefix=f"dim-({Args.latent_dim})/manhattan-{key}/n-({n})-n_r-({neighbor_r})/r-({r})/lr-({Args.lr})")
            plan2vec(f"manhattan-{key}", steps=n, success_r=1.2e-4 * float(r),
                     neighbor_r=neighbor_r,
                     prefix=f"dim-({Args.latent_dim})/manhattan-{key}/n-({n})-n_r-({neighbor_r})/r-({r})/lr-({Args.lr})")

    jaynes.listen()
