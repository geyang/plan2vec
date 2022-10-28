from plan2vec_experiments import instr
from plan2vec.plan2vec.plan2vec_state import DEBUG, Args, main
import jaynes


def common_config():
    local_metric_exp_path = "episodeyang/plan2vec/2019/05-05/c-maze/c_maze_local_metric_sweep/13.29/01.187654"

    Args.env_id = "CMazeDiscreteIdLess-v0"
    Args.num_epochs = 5000
    Args.lr = 3e-4
    Args.gamma = 1
    Args.target_update = 0.9
    Args.visualization_interval = 10
    Args.top_k = 20
    Args.H = 20
    Args.r_scale = 0.2

    Args.optim_batch_size = 128
    Args.optim_steps = 50
    Args.num_envs = 100

    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


def plan2vec():
    Args.binary_reward = False
    DEBUG.supervised_value_fn = False
    DEBUG.real_r_distance = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


def plan2vec_binary_reward():
    Args.binary_reward = True
    DEBUG.supervised_value_fn = False
    DEBUG.random_policy = False
    DEBUG.oracle_eps_greedy = False
    DEBUG.real_r_distance = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


def plan2vec_supervised_value_function():
    from ml_logger import logger
    Args.num_epochs = 100
    Args.binary_reward = None
    DEBUG.supervised_value_fn = True
    DEBUG.random_policy = False
    DEBUG.oracle_eps_greedy = False
    DEBUG.real_r_distance = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    logger.log_text("""
    # Supervised Value
    
    Should get very bad results
    """, filename="README.md")
    jaynes.run(_)


def plan2vec_real_r():
    Args.binary_reward = None
    DEBUG.supervised_value_fn = False
    DEBUG.oracle_eps_greedy = False
    DEBUG.random_policy = False
    DEBUG.real_r_distance = True
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


def plan2vec_oracle_eps_greedy():
    Args.binary_reward = None
    DEBUG.supervised_value_fn = False
    DEBUG.oracle_eps_greedy = True
    DEBUG.random_policy = False
    DEBUG.real_r_distance = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)

def plan2vec_random_policy():
    Args.binary_reward = None
    DEBUG.supervised_value_fn = False
    DEBUG.oracle_eps_greedy = False
    DEBUG.random_policy = True
    DEBUG.real_r_distance = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)

if __name__ == "__main__":
    # jaynes.config("local")
    # plan2vec()
    # exit()

    jaynes.config("vector")

    common_config()
    for Args.eps_greedy in [0.1, 0.4]:
        plan2vec_real_r()
        # plan2vec_supervised_value_function()
        plan2vec()
        plan2vec_binary_reward()

    jaynes.listen()
