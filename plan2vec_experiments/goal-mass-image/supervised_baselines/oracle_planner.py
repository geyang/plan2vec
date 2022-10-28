from plan2vec_experiments import instr, config_dash
from plan2vec.plan2vec.plan2vec_img import DEBUG, Args, main
import jaynes


def common_config():
    Args.start_seed = 5 * 100

    Args.num_epochs = 200
    Args.lr = 3e-5
    Args.gamma = 0.97
    Args.target_update = 0.9
    Args.top_k = None
    Args.plan_steps = 1
    Args.H = 30
    Args.r_scale = 0.2
    # Args.n_rollouts = 100

    Args.optim_epochs = 32

    Args.env_id = "GoalMassDiscreteImgIdLess-v0"
    local_metric_exp_path = "episodeyang/plan2vec/2019/05-27/goal-mass-image/local_metric/17.13/59.859207"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


def plan2vec_oracle_planner():
    Args.binary_reward = None
    DEBUG.oracle_planning = True
    DEBUG.supervised_value_fn = False
    Args.term_r, DEBUG.ground_truth_success = 0.04, True
    DEBUG.ground_truth_neighbor_r = 0.04
    DEBUG.real_r_distance = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


if __name__ == "__main__":
    jaynes.config("vector-gpu")

    common_config()

    config_dash("""
    ---
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
    - Args.n_rollouts
    - {metrics: success_rate/mean}
    - DEBUG.ground_truth_success
    - DEBUG.ground_truth_neighbor_r
    """)

    # for 400 rollouts, the visualization takes too long. So we cap it at 300.
    for Args.n_rollouts in [100, 200]:
        plan2vec_oracle_planner()

    jaynes.listen()
