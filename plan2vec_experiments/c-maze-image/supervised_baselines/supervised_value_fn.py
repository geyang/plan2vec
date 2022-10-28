from plan2vec_experiments import instr, config_dash
from plan2vec.plan2vec.plan2vec_img import DEBUG, Args, main
import jaynes


def common_config():

    Args.start_seed = 5 * 100

    Args.num_epochs = 10000
    Args.lr = 3e-5
    Args.gamma = 0.97
    Args.target_update = 0.9
    Args.top_k = None
    Args.plan_steps = 1
    Args.H = 30
    Args.r_scale = 0.2
    Args.n_rollouts = 200

    Args.optim_epochs = 32

    Args.env_id = "CMazeDiscreteImgIdLess-v0"
    local_metric_exp_path = "episodeyang/plan2vec/2019/05-27/c-maze-image/c_maze_img_local_metric/16.31/59.431952"
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
    jaynes.config("vector-gpu")

    common_config()

    config_dash("""
    charts:
    - {yKey: loss/mean, xKey: epoch}
    - {yKey: success_rate/mean, xKey: epoch}
    - {glob: '**/*connected.png', type: file}
    - {glob: '**/*neighbor_states.png', type: file}
    - {yKey: debug/supervised_rg/mean}
    - {yKey: debug/supervised_rg/min}
    - {yKey: debug/supervised_rg/max}
    - {yKey: true_delta/mean}
    keys:
    - run.status
    - Args.lr
    - {metrics: success_rate/mean}
    - DEBUG.ground_truth_success
    - DEBUG.ground_truth_neighbor_r
    """)

    for Args.lr in [3e-5]:  # , 1e-4, 3e-4, 1e-3]:
        plan2vec_supervised_value_function()

    jaynes.listen()
