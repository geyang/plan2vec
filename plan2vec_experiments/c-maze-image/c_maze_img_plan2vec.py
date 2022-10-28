def common_config():
    Args.num_epochs = 5000
    Args.lr = 3e-4
    Args.gamma = 1
    Args.target_update = 0.9
    Args.top_k = 30  # remove
    Args.H = 50
    Args.r_scale = 0.2

    Args.n_rollouts = 200

    # Args.optim_batch_size = 128
    # Args.optim_epochs = 50
    # Args.batch_n = 100

    Args.env_id = "CMazeDiscreteImgIdLess-v0"
    local_metric_exp_path = "episodeyang/plan2vec/2019/05-07/c-maze-image/c_maze_local_metric/21.19/10.332015"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


def plan2vec(**_):
    common_config()

    Args.binary_reward = False
    DEBUG.supervised_value_fn = False
    DEBUG.random_policy = False
    DEBUG.oracle_eps_greedy = False
    DEBUG.real_r_distance = False

    Args.update(_)

    _ = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


def plan2vec_eps(eps):
    common_config()

    Args.eps_greedy = eps
    Args.binary_reward = False
    DEBUG.supervised_value_fn = False
    DEBUG.random_policy = False
    DEBUG.oracle_eps_greedy = False
    DEBUG.real_r_distance = False
    _ = instr(main, __postfix=f"eps_{eps:0.2f}", **vars(Args), _DEBUG=vars(DEBUG))
    config_charts(path="sweep_eps.charts.yml")

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
    from plan2vec_experiments import instr, config_charts
    from plan2vec.plan2vec.plan2vec_img import DEBUG, Args, main
    import jaynes

    jaynes.config("vector-gpu")

    for global_metric in ['ResNet18L2', 'ResNet18Kernel', 'ResNet18CoordAsymmetricL2', 'ResNet18AsymmetricL2']:
        plan2vec(global_metric=global_metric)

    # plan2vec_real_r()
    # plan2vec_supervised_value_function()
    # plan2vec()
    # plan2vec_binary_reward()
    # for eps in [0.05, 0.1, 0.3, 0.5, 0.8, 1]:
    #     plan2vec_eps(eps)

    jaynes.listen()
