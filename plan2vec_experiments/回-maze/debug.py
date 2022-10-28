from plan2vec_experiments import instr
from plan2vec.plan2vec.plan2vec_state import DEBUG, Args, main
import jaynes


def common_config():
    Args.env_id = "回MazeDiscreteIdLess-v0"
    Args.num_epochs = 5000

    Args.n_rollouts = 200
    Args.lr = 1e-4
    Args.gamma = 1
    Args.target_update = 0.95
    Args.visualization_interval = 10
    Args.top_k = 50
    Args.H = 50
    Args.r_scale = 0.2

    # Args.optim_batch_size = 128
    # Args.optim_steps = 50
    # Args.batch_n = 100

    local_metric_exp_path = "episodeyang/plan2vec/2019/05-13/回-maze/回_maze_local_metric/13.06/48.004024"
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
    from ml_logger import logger

    Args.num_epochs = 100
    Args.binary_reward = None
    DEBUG.supervised_value_fn = False
    DEBUG.oracle_eps_greedy = True
    DEBUG.random_policy = False
    DEBUG.real_r_distance = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    logger.log_text("""
    # Supervised Value
    
    Should fail. Test with eps-greedy
    """, filename="README.md")
    jaynes.run(_)


def plan2vec_random_policy():
    from ml_logger import logger

    Args.num_epochs = 100

    Args.binary_reward = None
    DEBUG.supervised_value_fn = False
    DEBUG.oracle_eps_greedy = False
    DEBUG.random_policy = True
    DEBUG.real_r_distance = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    logger.log_text("""
    # Random Policy Baseline
    
    Should fail. Test with eps-greedy
    """, filename="README.md")
    jaynes.run(_)


if __name__ == "__main__":
    jaynes.config("local")

    DEBUG.ground_truth_neighbor_r = 0.1
    common_config()

    for Args.lr in [3e-4, 1e-4]:
        for Args.eps_greedy in [0.05, 0.1]:
            plan2vec()
            exit()
            plan2vec_binary_reward()

    # for Args.eps_greedy in [0.1, 0.2, 0.3]:
    #     for Args.target_update in [0.95, 0.97, 0.99, 5, 10, 20, 40]:
    #         for Args.lr in [3e-4, 1e-4]:
    #             for Args.eps_greedy in [0.05, 0.1]:
    #                 # plan2vec_real_r()
    #                 # plan2vec_supervised_value_function()
    #                 plan2vec()
    #                 plan2vec_binary_reward()

    # plan2vec_supervised_value_function()
    # # sleep(0.1)
    #
    # for Args.eps_greedy in [0.05, 0.1]:
    #     plan2vec_oracle_eps_greedy()
    #     # sleep(0.1)
    #
    # plan2vec_random_policy()

    jaynes.listen()
