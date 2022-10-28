from plan2vec_experiments import instr
from ml_logger import logger
from plan2vec.plan2vec.plan2vec_state import DEBUG, Args, main
import jaynes


def common_config():
    local_metric_exp_path = "episodeyang/plan2vec/2019/04-25/local_metric_state/12.28/36.983169"

    Args.num_epochs = 2000
    Args.lr = 3e-5
    Args.gamma = 0.97 / 20
    Args.target_update = 0.9
    Args.visualization_interval = 10
    Args.top_k = 20
    Args.H = 20

    Args.optim_batch_size = 128
    Args.optim_steps = 50
    Args.num_envs = 100

    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


def plan2vec():
    common_config()
    Args.binary_reward = False
    DEBUG.supervised_value_fn = False
    DEBUG.real_r_distance = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


def plan2vec_binary_reward():
    common_config()
    Args.binary_reward = True
    DEBUG.supervised_value_fn = False
    DEBUG.real_r_distance = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


def plan2vec_supervised_value_function():
    common_config()
    Args.binary_reward = None
    DEBUG.supervised_value_fn = True
    DEBUG.real_r_distance = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


def plan2vec_real_r():
    common_config()
    Args.binary_reward = None
    DEBUG.supervised_value_fn = False
    DEBUG.real_r_distance = True
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


if __name__ == "__main__":
    # jaynes.config("local")
    # plan2vec()
    # exit()

    for Args.r_scale in [1 / 20., 1 / 40., 1 / 5., 5]:
        jaynes.config("vector")
        plan2vec()

        jaynes.config("vector")
        plan2vec_binary_reward()

        jaynes.config("vector")
        plan2vec_supervised_value_function()

        jaynes.config("vector")
        plan2vec_real_r()

    jaynes.listen()
