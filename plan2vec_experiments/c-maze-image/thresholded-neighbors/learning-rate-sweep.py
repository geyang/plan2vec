from plan2vec_experiments import instr


def common_config():
    Args.env_id = "CMazeDiscreteImgIdLess-v0"
    Args.num_epochs = 5000
    Args.lr = 3e-4
    Args.gamma = 1
    Args.target_update = 0.9
    Args.top_k = None
    Args.H = 50
    Args.r_scale = 0.2

    Args.n_rollouts = 100

    # Args.optim_batch_size = 128
    # Args.optim_epochs = 50
    # Args.batch_n = 100

    local_metric_exp_path = "episodeyang/plan2vec/2019/05-07/c-maze-image/c_maze_local_metric/21.19/10.332015"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


def plan2vec():
    common_config()
    Args.binary_reward = False
    DEBUG.supervised_value_fn = False
    DEBUG.random_policy = False
    DEBUG.oracle_eps_greedy = False
    DEBUG.real_r_distance = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


def plan2vec_binary_reward():
    common_config()
    Args.binary_reward = True
    DEBUG.supervised_value_fn = False
    DEBUG.random_policy = False
    DEBUG.oracle_eps_greedy = False
    DEBUG.real_r_distance = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


def plan2vec_supervised_value_function():
    from ml_logger import logger
    common_config()
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
    common_config()
    Args.binary_reward = None
    DEBUG.supervised_value_fn = False
    DEBUG.oracle_eps_greedy = False
    DEBUG.random_policy = False
    DEBUG.real_r_distance = True
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


def plan2vec_oracle_eps_greedy():
    common_config()
    Args.binary_reward = None
    DEBUG.supervised_value_fn = False
    DEBUG.oracle_eps_greedy = True
    DEBUG.random_policy = False
    DEBUG.real_r_distance = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


def plan2vec_random_policy():
    common_config()
    Args.binary_reward = None
    DEBUG.supervised_value_fn = False
    DEBUG.oracle_eps_greedy = False
    DEBUG.random_policy = True
    DEBUG.real_r_distance = False
    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


if __name__ == "__main__":
    from plan2vec.plan2vec.plan2vec_img import DEBUG, Args, main
    import jaynes

    jaynes.config("vector-gpu")
    # plan2vec_real_r()
    # plan2vec_supervised_value_function()
    for Args.lr in [1e-5, 3e-5, 1e-4, 3e-4]:
        for Args.start_seed in range(3):
            plan2vec()
            plan2vec_binary_reward()
    # plan2vec_random_policy()
    jaynes.listen()
