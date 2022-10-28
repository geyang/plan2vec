from plan2vec_experiments import instr, config_charts
from plan2vec.plan2vec.plan2vec_streetlearn import DEBUG, Args, main
import jaynes


def common_config():
    Args.seed = 5 * 100

    Args.num_epochs = 1000
    Args.lr = 3e-5
    Args.gamma = 0.97
    Args.target_update = 0.9
    Args.top_k = None
    Args.plan_steps = 1
    Args.H = 50
    Args.r_scale = 0.2

    Args.optim_epochs = 32
    assert Args.batch_n == 20
    assert Args.H == 50

    # make this one to see early stage to make sure
    Args.visualization_interval = 10
    # turn off checkpointing b/c models are large
    Args.checkpoint_interval = None

    DEBUG.pretrain_global = False

    # Args.data_path = None
    local_metric_exp_path = "episodeyang/plan2vec/2019/06-20/streetlearn/local_metric/23.19/07.247751"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric_400.pkl"


def plan2vec_supervised_value_fn(dataset, prefix):
    Args.data_path = f"~/fair/streetlearn/processed-data/{dataset}"

    Args.binary_reward = None
    assert DEBUG.oracle_planning is False
    DEBUG.supervised_value_fn = True
    # todo: Need to add scalar
    Args.term_r, DEBUG.ground_truth_success = 2e-4, True
    DEBUG.ground_truth_neighbor_r = 2e-4
    assert DEBUG.real_r_distance is False
    _ = instr(main, __postfix=prefix, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


def exp_sweep(sig, exp, base=10):
    """
    Exponentially sweep through a large number of parameters.

    :param sig: array of [1, 3, 9] etc, the significand of the floating point number.
    :param exp: array of [-6, -5 ...]. The exponent. Need to be float otherwise numpy erros out for negative power.
    :param base: default to 10, can be other values.
    :return: a flattened numpy array.
    """
    _ = np.array(sig)[None, :] * np.power(base, exp)[:, None]
    return _.flatten()


if __name__ == "__main__":
    import numpy as np

    jaynes.config("vector-gpu")

    common_config()

    for Args.global_metric in ['ResNet18L2',
                               'GlobalMetricConvL2_s1',
                               'GlobalMetricConvDeepL2',
                               'GlobalMetricConvDeepL2_wide']:
        for key in ['tiny', 'small']:
            for Args.lr in exp_sweep([1, 3, 6, 8], exp=[-6., -5., -4., -3.]):
                plan2vec_supervised_value_fn(f"manhattan-{key}", Args.global_metric)
                config_charts()

    jaynes.listen()
