from plan2vec_experiments import instr, config_charts
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

    Args.env_id = "CMazeDiscreteImgIdLess-v0"
    local_metric_exp_path = "geyang/plan2vec/2019/06-21/c-maze-image/c_maze_img_local_metric/22.39/28.297534"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


# noinspection PyShadowingNames
def plan2vec_gt_neighbor_binary(seed):
    Args.start_seed = seed
    Args.binary_reward = True
    DEBUG.real_r_distance = False
    DEBUG.supervised_value_fn = False
    Args.term_r, DEBUG.ground_truth_success = 0.04, True
    # DEBUG.ground_truth_neighbor_r = 0.04
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG), __postfix=f"binary/{Args.global_metric}/lr-({Args.lr})")
    config_charts()
    jaynes.run(_)


# note: non-binary version need to use the real-r-distance.
# noinspection PyShadowingNames
def plan2vec_gt_neighbor_real_r(seed):
    Args.start_seed = seed
    Args.binary_reward = False
    DEBUG.real_r_distance = True
    DEBUG.supervised_value_fn = False
    Args.term_r, DEBUG.ground_truth_success = 0.04, True
    # DEBUG.ground_truth_neighbor_r = 0.04
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG), __postfix=f"dense/{Args.global_metric}/lr-({Args.lr})")
    config_charts()
    jaynes.run(_)


if __name__ == "__main__":
    jaynes.config("vector-gpu")

    common_config()

    for Args.global_metric in [
        'ResNet18L2',
        'ResNet18CoordL2',
        'GlobalMetricCoordConvL2',
        'GlobalMetricConvL2_s1'
    ]:
        for Args.lr in [1e-4, 3e-4, 6e-4, 8e-4, 1e-3, 3e-3, 1e-2, ]:
            for seed in range(4):
                plan2vec_gt_neighbor_binary(seed * 100)

        for Args.lr in [1e-4, 3e-4, 6e-4, 8e-4, 1e-3, 3e-3, 1e-2, ]:
            for seed in range(4):
                plan2vec_gt_neighbor_real_r(seed * 100)

    jaynes.listen()
