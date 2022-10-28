from plan2vec_experiments import instr, config_dash
from plan2vec.plan2vec.plan2vec_img import DEBUG, Args, main
import jaynes


def common_config():
    Args.start_seed = 5 * 100

    Args.num_epochs = 5000
    Args.lr = 3e-5
    Args.gamma = 0.97
    Args.target_update = 0.9
    Args.top_k = None
    Args.plan_steps = 1
    Args.H = 30
    Args.r_scale = 0.2
    Args.n_rollouts = 200

    Args.optim_epochs = 32

    # turn off checkpointing b/c models are large
    Args.checkpoint_interval = None

    DEBUG.pretrain_global = True
    DEBUG.pretrain_num_epochs = 100_000
    DEBUG.pretrain_viz_interval = 20

    Args.env_id = "GoalMassDiscreteImgIdLess-v0"
    local_metric_exp_path = "episodeyang/plan2vec/2019/05-27/goal-mass-image/local_metric/17.13/59.859207"
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
    - {xKey: epoch, yKey: loss/mean}
    - xKey: pretrain/epoch
      yKey: pretrain/loss
    - xKey: epoch
      yDomain: [0, 1]
      yKey: success_rate/mean
    - {glob: '**/embed_2d*.png', type: file}
    - {glob: '**/*connected.png', type: file}
    - {glob: '**/*neighbor_states.png', type: file}
    - yDomain: [0, 0.03]
      yKey: true_delta/mean
    - {yKey: dt_epoch}
    keys:
    - run.status
    - Args.n_rollouts
    - DEBUG.global_pretrain
    - DEBUG.ground_truth_success
    - DEBUG.ground_truth_neighbor_r
    - Args.lr
    - Args.loss
    - {metrics: success_rate/mean}
    - Args.model
    """)

    for Args.lr in [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]:
        plan2vec_supervised_value_function()

    jaynes.listen()
