from plan2vec_experiments import instr
from plan2vec.plan2vec.plan2vec_state import DEBUG, Args, main
import jaynes


def common_config():
    Args.env_id = "回MazeDiscreteIdLess-v0"
    Args.num_epochs = 5000

    Args.n_rollouts = 200
    Args.lr = 1e-4
    Args.gamma = 1
    Args.target_update = 0.95
    Args.visualization_interval = 10
    Args.top_k = None
    Args.H = 50
    Args.r_scale = 0.2

    # Args.optim_batch_size = 128
    # Args.optim_steps = 50
    # Args.batch_n = 100

    local_metric_exp_path = "episodeyang/plan2vec/2019/05-18/回-maze/回_maze_local_metric/13.19/21.629282"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


def plan2vec():
    Args.binary_reward = False
    DEBUG.supervised_value_fn = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


def plan2vec_binary_reward():
    Args.binary_reward = True
    DEBUG.supervised_value_fn = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


def plan2vec_supervised_value_function():
    Args.binary_reward = None
    DEBUG.supervised_value_fn = True
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


if __name__ == "__main__":
    jaynes.config("vector")

    common_config()

    for Args.lr in [3e-4, 1e-4]:
        for Args.eps_greedy in [0.05, 0.1]:
            plan2vec()
            plan2vec_binary_reward()

    # plan2vec_supervised_value_function()

    jaynes.listen()
