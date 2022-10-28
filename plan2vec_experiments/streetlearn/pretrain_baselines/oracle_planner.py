from plan2vec_experiments import instr, config_charts
from plan2vec.plan2vec.plan2vec_streetlearn_2 import DEBUG, Args, main
import jaynes


def common_config():
    Args.seed = 5 * 100

    Args.num_epochs = 500
    Args.lr = 3e-5
    Args.gamma = 0.97
    Args.target_update = 0.9
    Args.top_k = None
    Args.plan_steps = 1
    Args.H = 50
    Args.r_scale = 0.2

    # Args.optim_batch_size = 128
    # Args.optim_epochs = 50
    # Args.batch_n = 100

    # Args.data_path = None
    local_metric_exp_path = "episodeyang/plan2vec/2019/06-20/streetlearn/local_metric/23.19/07.247751"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric_400.pkl"


def plan2vec_oracle_planner(dataset, prefix, goal_r=None):
    Args.data_path = f"~/fair/streetlearn/processed-data/{dataset}"
    Args.binary_reward = True

    DEBUG.value_fn_pretrain_goal_r = goal_r
    DEBUG.oracle_planning = True

    Args.optim_epochs = 0
    DEBUG.supervised_value_fn = False
    Args.term_r, DEBUG.ground_truth_success = 2e-4, True
    DEBUG.ground_truth_neighbor_r = 2e-4
    DEBUG.real_r_distance = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG), __postfix=prefix)
    config_charts()
    jaynes.run(_)


if __name__ == "__main__":
    jaynes.config("vector-gpu")

    common_config()

    # todo: loop through DEBUG.goal_r
    for key in ['tiny', 'small', 'medium', 'large', 'xl']:
        plan2vec_oracle_planner(f"manhattan-{key}", f"manhattan-{key}")

    jaynes.listen()
