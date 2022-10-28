from plan2vec_experiments import instr
from plan2vec.plan2vec.plan2vec_state import DEBUG, Args, main
import jaynes


def common_config():
    Args.env_id = "CMazeDiscreteIdLess-v0"
    Args.num_epochs = 5000

    Args.n_rollouts = 200
    Args.lr = 1e-4
    Args.gamma = 1
    Args.target_update = 0.95
    Args.visualization_interval = 10
    Args.top_k = 50
    Args.H = 50
    Args.r_scale = 0.2

    Args.optim_steps = 32

    # Args.optim_batch_size = 128
    # Args.optim_steps = 50
    # Args.batch_n = 100

    local_metric_exp_path = "episodeyang/plan2vec/2019/05-14/c-maze/c_maze_local_metric/14.56/08.906783"
    # local_metric_exp_path = "episodeyang/plan2vec/2019/05-05/c-maze/c_maze_local_metric_sweep/13.29/01.187654"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


def plan2vec_binary_reward():
    Args.binary_reward = True
    DEBUG.supervised_value_fn = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


if __name__ == "__main__":
    jaynes.config("vector")

    common_config()
    DEBUG.ground_truth_neighbor_r = 0.04

    for Args.lr in [3e-4, 1e-4]:
        for Args.eps_greedy in [0.05, 0.1]:
            plan2vec_binary_reward()

    jaynes.listen()
