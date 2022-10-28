from plan2vec.plotting.maze_world.connect_the_dots_state_maze import Args, main

if __name__ == "__main__":
    import jaynes
    from plan2vec_experiments import instr, RUN
    from os.path import join as pJoin
    from ml_logger import logger

    jaynes.config('vector')

    Args.latent_dim = 32
    Args.env_id = "GoalMassDiscreteIdLess-v0"

    logger.configure(log_directory=RUN.server, register_experiment=False)

    # prefix = "/episodeyang/plan2vec/2019/05-22/c-maze/c_maze_local_metric"
    prefix = "/episodeyang/plan2vec/2019/05-10/goal-mass-image/local_metric/14.15/38.768056"
    weight_paths = logger.glob("**/models/local_metric.pkl", prefix)

    for path in weight_paths:
        Args.load_local_metric = pJoin(prefix, path)
        jaynes.run(instr(main, **vars(Args)))

    jaynes.listen()
