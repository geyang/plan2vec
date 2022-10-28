from plan2vec.plotting.maze_world.connect_the_dots_image_maze import Args, main

if __name__ == "__main__":
    import jaynes
    from plan2vec_experiments import instr, RUN, config_charts, dir_prefix
    from os.path import join as pJoin, dirname
    from ml_logger import logger

    jaynes.config('vector-gpu')

    logger.configure(log_directory=RUN.server, register_experiment=False)

    kwargs = []
    with logger.PrefixContext(dir_prefix()):
        weight_paths = logger.glob("**/models/local_metric.pkl")
        logger.print(*weight_paths, sep="\n")
        for p in weight_paths:
            parameter_path = pJoin(dirname(dirname(p)), 'parameters.pkl')
            env_id = logger.get_parameters('Args.env_id', path=parameter_path, default=None) \
                     or logger.get_parameters('local_metric.env_id', path=parameter_path, default=None)

            kwargs.append(dict(env_id=env_id, load_local_metric=logger.abspath(p)))

    for _ in kwargs:
        jaynes.run(instr(main, n_rollouts=100, **_))
        config_charts()

    jaynes.listen()
