from plan2vec_experiments import instr
from plan2vec.plan2vec.plan2vec_img import DEBUG, Args, main
import jaynes


def common_config():
    Args.env_id = "CMazeDiscreteImgIdLess-v0"

    Args.start_seed = 5 * 100

    Args.num_epochs = 10000
    Args.lr = 3e-5
    Args.gamma = 0.97
    Args.target_update = 0.9
    Args.top_k = None
    Args.plan_steps = 2
    Args.H = 20
    Args.r_scale = 0.2
    Args.n_rollouts = 200

    Args.optim_epochs = 32

    local_metric_exp_path = "episodeyang/plan2vec/2019/05-20/c-maze-image/c_maze_img_local_metric/23.57/22.940831"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"




def plan2vec_supervised_value_function():
    Args.binary_reward = None
    DEBUG.supervised_value_fn = True
    Args.term_r, DEBUG.ground_truth_success = 0.04, True
    DEBUG.ground_truth_neighbor_r = 0.04
    DEBUG.real_r_distance = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


if __name__ == "__main__":
    jaynes.config("vector")

    common_config()

    plan2vec_supervised_value_function()

    jaynes.listen()
