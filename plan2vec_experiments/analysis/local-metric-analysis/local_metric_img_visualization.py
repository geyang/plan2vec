from plan2vec.plotting.maze_world.connect_the_dots_image_maze import Args, main

if __name__ == "__main__":
    import jaynes
    from plan2vec_experiments import instr, RUN, config_charts, dir_prefix
    from os.path import join as pJoin, dirname
    from ml_logger import logger

    logger.configure(log_directory=RUN.server, register_experiment=False)

    # glob_root = dir_prefix()
    glob_root = "/geyang/plan2vec/2019/12-16/analysis/local-metric-analysis/all_local_metric"

    kwargs = []
    with logger.PrefixContext(glob_root):
        # note: rope uses {}-{} as postfix. maze do not.
        weight_paths = logger.glob("**/models/local_metric.pkl")
        logger.print(*weight_paths, sep="\n")
        for p in weight_paths:
            parameter_path = pJoin(dirname(dirname(p)), 'parameters.pkl')
            env_id = logger.get_parameters('Args.env_id', path=parameter_path, default=None)
            logger.abspath(p)

            kwargs.append(dict(env_id=env_id, load_local_metric=logger.abspath(p)))

    jaynes.config()
    for _ in kwargs:
        jaynes.run(instr(main, n_rollouts=100, **_))
        config_charts("""
        charts:
          - type: file
            glob: "**/*render.png"
          - type: file
            glob: "**/*data.png"
          - type: file
            glob: "**/*connected.png"
          - type: file
            glob: "**/*gt.png"
          - type: file
            glob: "**/*gt_wider.png"
        keys:
          - run.status
          - Args.env_id
          - Args.load_local_metric
        """)

    jaynes.listen()
