from plan2vec_experiments import instr
from plan2vec.plan2vec.plan2vec_streetlearn import DEBUG, Args, main
import jaynes


def common_config():
    Args.seed = 5 * 100

    Args.num_epochs = 5000
    Args.lr = 3e-5
    Args.gamma = 0.97
    Args.target_update = 0.9
    Args.top_k = None
    Args.plan_steps = 1
    Args.H = 50
    Args.r_scale = 0.2

    Args.data_path = "~/fair/streetlearn/processed-data/manhattan-medium"

    # Args.optim_batch_size = 128
    # Args.optim_epochs = 50
    # Args.batch_n = 100

    local_metric_exp_path = "episodeyang/plan2vec/2019/05-20/streetlearn/local_metric/18.12/08.412698"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric_100.pkl"


def plan2vec_binary_reward():
    Args.binary_reward = True
    DEBUG.real_r_distance = False
    DEBUG.ground_truth_neighbor_r = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


def plan2vec_binary_reward_real_neighbor():
    Args.binary_reward = True
    DEBUG.real_r_distance = False
    DEBUG.ground_truth_neighbor_r = 2e-4
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


def plan2vec_real_r():
    Args.binary_reward = None
    DEBUG.real_r_distance = True
    DEBUG.ground_truth_neighbor_r = 2e-4
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


if __name__ == "__main__":
    jaynes.config("vector-gpu", runner=dict(n_cpu=40))

    common_config()

    for Args.r_scale in [0.1, 0.3]:
        for Args.lr in [3e-4, 4e-4]:
            for Args.eps_greedy in [0.05, 0.1, 0.15, 0.2]:
                plan2vec_binary_reward_real_neighbor()

    jaynes.listen()
