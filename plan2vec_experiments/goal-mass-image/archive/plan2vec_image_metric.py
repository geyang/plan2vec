from plan2vec_experiments import instr
from plan2vec.plan2vec.plan2vec_img import DEBUG, Args, main
import jaynes


def common_config():
    Args.start_seed = 5 * 100

    Args.num_epochs = 5000
    Args.lr = 3e-5
    Args.gamma = 0.97
    Args.target_update = 0.9
    Args.top_k = 100
    Args.H = 20
    Args.r_scale = 0.2

    Args.optim_batch_size = 128
    Args.optim_epochs = 50
    Args.batch_n = 100

    local_metric_exp_path = "episodeyang/plan2vec/2019/04-30/goal-mass-image/local_metric/13.04/50.045261"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"
    # Args.load_pairwise_ds = "/episodeyang/plan2vec/2019/04-30/goal-mass-image/pairwise_map_reduce/15.55/26.201868"
    # Args.load_pairwise_ds = "/episodeyang/plan2vec/2019/04-30/goal-mass-image/pairwise_map_reduce/16.16/54.604432"


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
    import time

    jaynes.config("vector-gpu")

    common_config()
    for Args.r_scale in [1 / 40, 1 / 10, 2, 5]:
        for Args.lr in [1e-4, 3e-4, 1e-3, 3e-3]:
            plan2vec()
            time.sleep(0.1)
            plan2vec_binary_reward()
            time.sleep(0.1)

    jaynes.listen()
