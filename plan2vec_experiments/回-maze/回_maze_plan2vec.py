from plan2vec_experiments import instr, config_charts
from plan2vec.plan2vec.plan2vec_state import DEBUG, Args, main
import jaynes


def common_config():
    Args.num_epochs = 5000

    Args.lr = 3e-4
    Args.weight_decay = 0

    Args.gamma = 1
    Args.target_update = 0.9
    Args.top_k = None

    Args.neighbor_r = 1.
    Args.term_r = 1.
    Args.plan_steps = 1
    Args.H = 20
    Args.r_scale = 0.2

    Args.visualization_interval = 10

    Args.optim_epochs = 32
    # Args.optim_batch_size = 128
    # Args.optim_epochs = 50
    # Args.batch_n = 100

    Args.env_id = '回MazeDiscreteIdLess-v0'
    local_metric_exp_path = "geyang/plan2vec/2019/06-21/回-maze/回_maze_local_metric/20.47/27.023292"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


def plan2vec():
    common_config()
    Args.binary_reward = False
    DEBUG.supervised_value_fn = False
    DEBUG.real_r_distance = False
    _ = instr(main, __postfix="dense_reward", **vars(Args), _DEBUG=vars(DEBUG))
    config_charts()
    jaynes.run(_)


def plan2vec_binary_reward():
    common_config()
    Args.binary_reward = True
    DEBUG.supervised_value_fn = False
    DEBUG.real_r_distance = False
    _ = instr(main, __postfix="binary_reward", **vars(Args), _DEBUG=vars(DEBUG))
    config_charts()
    jaynes.run(_)


def plan2vec_supervised_value_function():
    common_config()
    Args.num_epochs = 20  # only need a few

    Args.binary_reward = None
    DEBUG.supervised_value_fn = True
    DEBUG.real_r_distance = False
    _ = instr(main, __postfix="supervised_value_fn", **vars(Args), _DEBUG=vars(DEBUG))
    config_charts()
    jaynes.run(_)


def plan2vec_real_r():
    common_config()
    Args.binary_reward = None
    DEBUG.supervised_value_fn = False
    DEBUG.real_r_distance = True
    _ = instr(main, __postfix="debug_real_r", **vars(Args), _DEBUG=vars(DEBUG))
    config_charts()
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
    # plan2vec_real_r()
    # plan2vec_supervised_value_function()
    # plan2vec_random_policy()

    jaynes.listen()
