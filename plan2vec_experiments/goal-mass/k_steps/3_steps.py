from plan2vec_experiments import instr
from plan2vec.plan2vec.plan2vec_state import DEBUG, Args, main
import jaynes


def common_config():
    Args.env_id = "GoalMassDiscreteIdLess-v0"
    Args.num_epochs = 5000

    Args.n_rollouts = 100
    Args.lr = 3e-4
    Args.gamma = 1
    Args.target_update = 0.9
    Args.visualization_interval = 10
    Args.top_k = None
    Args.H = 20
    Args.r_scale = 0.2

    # Args.optim_batch_size = 128
    # Args.optim_epochs = 50
    # Args.batch_n = 100

    local_metric_exp_path = "episodeyang/plan2vec/2019/05-14/goal-mass/goal_mass_local_metric/14.56/07.432922"
    # local_metric_exp_path = "episodeyang/plan2vec/2019/04-29/goal-mass/goal_mass_local_metric/15.11/40.723973"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


def plan2vec():
    Args.binary_reward = False
    DEBUG.supervised_value_fn = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


def plan2vec_binary_reward():
    Args.binary_reward = True
    DEBUG.supervised_value_fn = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


def plan2vec_supervised_value_function():
    Args.binary_reward = None
    DEBUG.supervised_value_fn = True
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


if __name__ == "__main__":
    jaynes.config("vector")

    common_config()

    for Args.lr in [3e-4, 1e-4]:
        for Args.eps_greedy in [0.05, 0.1]:
            plan2vec()
            plan2vec_binary_reward()

    plan2vec_supervised_value_function()

    jaynes.listen()
