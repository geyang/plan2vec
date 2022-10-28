from params_proto.hyper import sweep


def common_config():
    Args.num_epochs = 5000
    Args.lr = 3e-4
    Args.gamma = 1
    Args.target_update = 0.9
    Args.top_k = 30  # remove
    Args.H = 50
    Args.r_scale = 0.2

    Args.n_rollouts = 200


def plan2vec(**_):
    common_config()

    Args.binary_reward = False
    DEBUG.supervised_value_fn = False
    DEBUG.random_policy = False
    DEBUG.oracle_eps_greedy = False
    DEBUG.real_r_distance = False

    Args.update(_)

    _ = instr(main, __postfix=f"{Args.env_id}/{Args.global_metric}", **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


if __name__ == "__main__":
    from plan2vec_experiments import instr, config_charts
    from plan2vec.plan2vec.plan2vec_img import DEBUG, Args, main
    import jaynes

    jaynes.config("vector-gpu")

    local_metric_paths = [
        "geyang/plan2vec/2019/06-21/goal-mass-image/goal_mass_img_local_metric/22.39/21.462733",
        "episodeyang/plan2vec/2019/05-10/回-maze-image/回_maze_img_local_metric/13.28/59.358543",
        "episodeyang/plan2vec/2019/05-07/c-maze-image/c_maze_local_metric/21.19/10.332015",
    ]

    with sweep.product(Args) as _Args:

        with sweep.zip(Args) as _Args:
            _Args.env_id = ["GoalMassDiscreteImgIdLess-v0", "回MazeDiscreteImgIdLess-v0", "CMazeDiscreteImgIdLess-v0"]
            _Args.load_local_metric = [f"/{key}/models/local_metric.pkl" for key in local_metric_paths]

        # _Args.global_metric = ['ResNet18L2', 'ResNet18Kernel', 'ResNet18CoordAsymmetricL2', 'ResNet18AsymmetricL2']
        _Args.global_metric = ['ResNet18L2', 'ResNet18CoordAsymmetricL2']

    for override in sweep:
        plan2vec(**override)

    jaynes.listen()
