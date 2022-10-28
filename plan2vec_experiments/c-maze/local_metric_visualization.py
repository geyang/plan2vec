from plan2vec.plotting.maze_world.connect_the_dots_state_maze import Args, main

if __name__ == "__main__":
    import jaynes
    from plan2vec_experiments import instr, RUN, dir_prefix, config_charts
    from os.path import join as pJoin, dirname
    from ml_logger import logger

    jaynes.config('vector')

    logger.configure(log_directory=RUN.server, register_experiment=False)

    kwargs = []
    with logger.PrefixContext(dir_prefix()):
        weight_paths = logger.glob("**/models/local_metric.pkl")
        logger.print(*weight_paths, sep="\n")
        for p in weight_paths:
            parameter_path = pJoin(dirname(dirname(p)), 'parameters.pkl')
            env_id = logger.get_parameters('Args.env_id', path=parameter_path, default=None) \
                     or logger.get_parameters('local_metric.env_id', path=parameter_path, default=None)
            print(env_id)

            kwargs.append(dict(env_id=env_id, load_local_metric=logger.abspath(p)))

    for _ in kwargs:
        jaynes.run(instr(main, **_))
        config_charts()

    jaynes.listen()

