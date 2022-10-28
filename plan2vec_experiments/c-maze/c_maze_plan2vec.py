from plan2vec.plan2vec.plan2vec_state import DEBUG, Args, main
from plan2vec_experiments import instr, config_charts, RUN
import jaynes


def common_config():
    Args.num_epochs = 5000

    Args.lr = 3e-4
    Args.eps_greedy = 0.05
    Args.weight_decay = 0

    Args.gamma = 1
    Args.target_update = 0.9
    Args.top_k = None

    Args.neighbor_r = 1.
    Args.term_r = 1.
    Args.plan_steps = 1
    Args.H = 50
    Args.r_scale = 0.2

    Args.visualization_interval = 10

    Args.optim_epochs = 32
    # Args.optim_batch_size = 128
    # Args.optim_steps = 50
    # Args.batch_n = 100

    Args.env_id = "CMazeDiscreteIdLess-v0"
    local_metric_exp_path = "geyang/plan2vec/2019/11-13/c-maze/c_maze_local_metric/02.11/16.628464"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


def plan2vec():
    common_config()

    Args.binary_reward = False
    DEBUG.supervised_value_fn = False
    DEBUG.real_r_distance = False
    _ = instr(main, __postfix="dense_reward", **vars(Args), _DEBUG=vars(DEBUG))
    config_charts()
    jaynes.run(_)


def plan2vec_eps(eps):
    common_config()

    Args.eps_greedy = eps
    Args.binary_reward = False
    DEBUG.supervised_value_fn = False
    DEBUG.real_r_distance = False
    _ = instr(main, __postfix=f"dense_reward_{eps:0.2f}_eps", **vars(Args), _DEBUG=vars(DEBUG))
    config_charts(path="sweep_eps.charts.yml")
    jaynes.run(_)


def plan2vec_binary_reward():
    common_config()

    Args.binary_reward = True
    DEBUG.supervised_value_fn = False
    DEBUG.random_policy = False
    DEBUG.oracle_eps_greedy = False
    DEBUG.real_r_distance = False
    _ = instr(main, __postfix="binary_reward", **vars(Args), _DEBUG=vars(DEBUG))
    config_charts()
    jaynes.run(_)


def plan2vec_supervised_value_function():
    common_config()

    from ml_logger import logger

    Args.num_epochs = 100
    Args.binary_reward = None
    DEBUG.supervised_value_fn = True
    DEBUG.random_policy = False
    DEBUG.oracle_eps_greedy = False
    DEBUG.real_r_distance = False
    _ = instr(main, __postfix="supervised_value_fn", **vars(Args), _DEBUG=vars(DEBUG))
    config_charts()
    logger.log_text("""
    # Supervised Value
    
    Should get very bad results
    """, filename="README.md")
    jaynes.run(_)


def plan2vec_real_r():
    common_config()

    Args.binary_reward = None
    DEBUG.supervised_value_fn = False
    DEBUG.oracle_eps_greedy = False
    DEBUG.random_policy = False
    DEBUG.real_r_distance = True
    _ = instr(main, __postfix="debug_real_r", **vars(Args), _DEBUG=vars(DEBUG))
    config_charts()
    jaynes.run(_)


def plan2vec_oracle_eps_greedy():
    from ml_logger import logger
    common_config()

    Args.num_epochs = 100
    Args.binary_reward = None
    DEBUG.supervised_value_fn = False
    DEBUG.oracle_eps_greedy = True
    DEBUG.random_policy = False
    DEBUG.real_r_distance = False
    _ = instr(main, __postfix="oracle_planner", **vars(Args), _DEBUG=vars(DEBUG))
    config_charts(path="oracle_planner.charts.yml")
    logger.log_text("""
    # Supervised Value
    
    Should fail. Test with eps-greedy
    """, filename="README.md")
    jaynes.run(_)


def plan2vec_random_policy():
    from ml_logger import logger
    common_config()

    Args.num_epochs = 100

    Args.binary_reward = None
    DEBUG.supervised_value_fn = False
    DEBUG.oracle_eps_greedy = False
    DEBUG.random_policy = True
    DEBUG.real_r_distance = False
    _ = instr(main, __postfix="random_policy", **vars(Args), _DEBUG=vars(DEBUG))
    config_charts(path="oracle_planner.charts.yml")
    logger.log_text("""
    # Random Policy Baseline
    
    Should fail. Test with eps-greedy
    """, filename="README.md", dedent=True)
    jaynes.run(_)


if __name__ == "__main__":
    jaynes.config("vector")

    plan2vec()
    # plan2vec_binary_reward()
    # plan2vec_supervised_value_function()
    # plan2vec_real_r()
    # plan2vec_random_policy()
    # plan2vec_oracle_eps_greedy()
    # for eps in [0.05, 0.1, 0.3, ]:
    #     plan2vec_eps(eps)

    jaynes.listen()
