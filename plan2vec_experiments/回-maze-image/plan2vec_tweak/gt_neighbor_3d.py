from plan2vec_experiments import instr, config_charts, dir_prefix, RUN
from plan2vec.plan2vec.plan2vec_img import DEBUG, Args, main
import jaynes


def common_config():
    Args.start_seed = 5 * 100
    Args.lr = 3e-5

    Args.gamma = 0.97
    Args.target_update = 0.9
    Args.top_k = None
    Args.plan_steps = 3
    Args.H = 30
    Args.r_scale = 0.2
    Args.n_rollouts = 400
    Args.timesteps = 2

    Args.latent_dim = 3

    # turn off checkpointing b/c models are large
    Args.checkpoint_interval = None

    Args.optim_batch_size = 128
    Args.optim_epochs = 32

    Args.num_epochs = 200  # we learn in 20 epochs..
    Args.visualization_interval = 1

    # DEBUG.global_pretrain = False
    # DEBUG.pretrain_num_epochs = 0
    # DEBUG.pretrain_viz_interval = None

    Args.env_id = "回MazeDiscreteImgIdLess-v0"
    # from ml_logger import logger
    #
    # logger.configure(RUN.server, dir_prefix(-2))
    # checkpoints = logger.glob("**/local_metric.pkl")
    # print(logger.prefix)
    # print(*checkpoints, sep="\n")
    #
    # Args.load_local_metric = checkpoints[0]
    local_metric_exp_path = "geyang/plan2vec/2019/06-21/回-maze-image/回_maze_img_local_metric/22.39/24.875458"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


def plan2vec_gt_neighbor_binary():
    Args.binary_reward = True
    DEBUG.real_r_distance = False
    DEBUG.supervised_value_fn = False
    Args.term_r, DEBUG.ground_truth_success = 0.04, True
    DEBUG.ground_truth_neighbor_r = 0.04
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


# note: non-binary version need to use the real-r-distance.
def plan2vec_gt_neighbor_real_r():
    Args.binary_reward = False
    DEBUG.real_r_distance = True
    DEBUG.supervised_value_fn = False
    Args.term_r, DEBUG.ground_truth_success = 0.04, True
    DEBUG.ground_truth_neighbor_r = 0.04
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


if __name__ == "__main__":
    jaynes.config("vector-gpu")

    common_config()

    for Args.global_metric in ['ResNet18L2', 'ResNet18CoordL2', 'GlobalMetricCoordConvL2', 'GlobalMetricConvL2_s1']:
        for Args.lr in [1e-4, 3e-4, 6e-4, 8e-4, 1e-3, 3e-3, 1e-2, ]:
            plan2vec_gt_neighbor_binary()
            config_charts()
        for Args.lr in [1e-4, 3e-4, 6e-4, 8e-4, 1e-3, 3e-3, 1e-2, ]:
            plan2vec_gt_neighbor_real_r()
            config_charts()

    jaynes.listen()
