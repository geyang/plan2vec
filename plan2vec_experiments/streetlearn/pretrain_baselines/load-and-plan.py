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

    Args.latent_dim = 2

    # make this one to see early stage to make sure
    Args.visualization_interval = 1
    # turn off checkpointing b/c models are large
    Args.checkpoint_interval = None
    Args.binary_reward = None

    Args.load_global_metric = "/geyang/plan2vec/2019/07-15/streetlearn/pretrain_baselines/" \
                              "sample-distribution-comparison/value-05-step-pretrain/dim-(2)/" \
                              "manhattan-medium/ResNet18L2/lr-(7.5e-08)/22.16/31.668959/" \
                              "value_fn_pretrain/global_metric_fn.pkl"

    # DEBUG.pretrain_num_epochs = 1

    # DEBUG.pretrain_global = True
    # # also try to pretrain with random sample,
    # # try sampling techniques
    # # add evaluation loops
    # DEBUG.value_fn_pretrain_global = True

    # Args.data_path = None
    local_metric_exp_path = "episodeyang/plan2vec/2019/06-20/streetlearn/local_metric/23.19/07.247751"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric_400.pkl"


def plan2vec_eval_planning(dataset, prefix, r):
    Args.data_path = f"~/fair/streetlearn/processed-data/{dataset}"

    assert DEBUG.pretrain_global is False
    assert DEBUG.value_fn_pretrain_global is False
    assert DEBUG.supervised_value_fn is False
    assert Args.binary_reward is None
    assert DEBUG.oracle_planning is False

    Args.term_r, DEBUG.ground_truth_success = 1.2e-4 * 20, True
    DEBUG.ground_truth_neighbor_r = r

    DEBUG.real_r_distance = False
    _ = instr(main, __postfix=f"eval-pretrained/{prefix}", **vars(Args), _DEBUG=vars(DEBUG))
    config_charts(path="plan2vec_eval_planning.charts.yml")
    jaynes.run(_)


def plan2vec_eval_control(dataset, prefix, neighbor_r, success_r):
    Args.data_path = f"~/fair/streetlearn/processed-data/{dataset}"

    Args.num_epochs = 20
    # freeze the learning todo: remove HER-relabeling and optimization.
    Args.lr = 0

    assert DEBUG.pretrain_global is False
    assert DEBUG.value_fn_pretrain_global is False
    assert DEBUG.supervised_value_fn is False
    assert Args.binary_reward is None
    assert DEBUG.oracle_planning is False

    DEBUG.ground_truth_neighbor_r = neighbor_r
    Args.term_r, DEBUG.ground_truth_success = success_r, True

    DEBUG.real_r_distance = False
    _ = instr(main, __postfix=f"eval-control/{prefix}", **vars(Args), _DEBUG=vars(DEBUG))
    config_charts(path="plan2vec_eval_control.charts.yml")
    jaynes.run(_)


if __name__ == "__main__":

    common_config()

    param_dict = {
        'ResNet18L2': {"lr": [1e-6, 3e-6, 1e-7, 3e-7], },
        # 'GlobalMetricConvL2_s1': {"lr": [1e-6, 3e-6, 6e-6]},
        # 'GlobalMetricConvDeepL2': {"lr": [1e-6, 3e-6, 6e-6]},
        # 'GlobalMetricConvDeepL2_wide': {"lr": [1e-6, 3e-6, 6e-6]}
    }

    # ResNet requires much less memory than the other.
    Args.global_metric = 'ResNet18L2'
    _ = param_dict['ResNet18L2']

    jaynes.config("vector-gpu")

    key = 'medium'

    # for n in [5, 10, 20, 40]:
    #     plan2vec_eval_planning(f"manhattan-{key}", r=1.2e-4 * float(n),
    #                            prefix=f"dim-({Args.latent_dim})/manhattan-{key}/r-({n})")
    #     plan2vec_eval_control(f"manhattan-{key}", r=1.2e-4 * float(n),
    #                           prefix=f"dim-({Args.latent_dim})/manhattan-{key}/r-({n})")

    for n in [5, 10, 20, 40]:
        for r in [10, 5, 1]:
            plan2vec_eval_control(f"manhattan-{key}", neighbor_r=1.2e-4 * float(n), success_r=1.2e-4 * float(r),
                                  prefix=f"dim-({Args.latent_dim})/manhattan-{key}/n-({n})/r-({r})")

    jaynes.listen()
