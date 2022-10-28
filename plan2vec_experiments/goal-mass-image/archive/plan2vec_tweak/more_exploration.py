from plan2vec_experiments import instr
from ml_logger import logger
from plan2vec.plan2vec.plan2vec_img import DEBUG, Args, main
import jaynes


def common_config():
    Args.start_seed = 5 * 100

    Args.num_epochs = 50000
    Args.lr = 3e-5
    Args.gamma = 0.97
    Args.target_update = 0.9
    Args.top_k = None
    Args.H = 20
    Args.r_scale = 0.2
    Args.n_rollouts = 200

    Args.optim_batch_size = 128
    Args.optim_epochs = 50
    Args.batch_n = 100

    local_metric_exp_path = "episodeyang/plan2vec/2019/05-10/goal-mass-image/local_metric/14.15/38.768056"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


def plan2vec():
    Args.binary_reward = False
    DEBUG.supervised_value_fn = False
    DEBUG.real_r_distance = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


def plan2vec_binary_reward():
    Args.binary_reward = True
    DEBUG.supervised_value_fn = False
    DEBUG.real_r_distance = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


if __name__ == "__main__":
    jaynes.config("vector-gpu")

    common_config()
    for Args.eps_greedy in [0.1, 0.2, 0.3, 0.4]:
        for Args.r_scale in [1 / 40, 1 / 10, 2, 5]:
            for Args.start_seed in range(5):
                plan2vec()
                plan2vec_binary_reward()

    jaynes.listen()
