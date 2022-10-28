"""
Use this one to check whether the model architecture is good for the learning problem.

Not all global architectures are good.
"""
from plan2vec_experiments import instr
from plan2vec.plan2vec.plan2vec_state import DEBUG, Args, main
import jaynes


def common_config():
    Args.num_epochs = 200
    Args.lr = 3e-4
    Args.gamma = 1
    Args.target_update = 0.9
    Args.visualization_interval = 10
    Args.top_k = None
    Args.neighbor_r = 1.
    Args.term_r = 1.
    Args.plan_steps = 1
    Args.H = 20
    Args.r_scale = 0.2

    Args.latent_dim = 50

    # Args.optim_batch_size = 128
    # Args.optim_epochs = 50
    # Args.batch_n = 100

    local_metric_exp_path = "episodeyang/plan2vec/2019/05-22/goal-mass/goal_mass_local_metric/16.07/42.861843"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


def plan2vec_supervised_value_function():
    Args.binary_reward = None
    DEBUG.supervised_value_fn = True
    DEBUG.real_r_distance = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


if __name__ == "__main__":

    jaynes.config("vector")

    common_config()
    
    for Args.lr in [1e-4, 3e-4, 1e-3]:
        plan2vec_supervised_value_function()

    jaynes.listen()
