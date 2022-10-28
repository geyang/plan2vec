from plan2vec.plotting.maze_world.connect_the_dots_image_maze import Args, main

if __name__ == "__main__":
    import jaynes
    from plan2vec_experiments import instr, RUN, config_charts, dir_prefix
    from os.path import join as pJoin, dirname, normpath
    from ml_logger import logger

    logger.configure(log_directory=RUN.server, register_experiment=False)

    # glob_root = dir_prefix()
    # glob_root = "/geyang/plan2vec/2019/12-16/analysis/local-metric-analysis/all_local_metric"
    # glob_root = "/geyang/plan2vec/2020/02-08/neo_plan2vec/uvpn_image/quick_eval_new_local_metric/local_metric/10.50"
    glob_root = "/geyang/plan2vec/2020/02-08/neo_plan2vec/uvpn_image/quick_eval_new_local_metric/local_metric/hige_loss/lr-sweep/12.24"

    kwargs = []
    with logger.PrefixContext(glob_root):
        # note: rope uses {}-{} as postfix. maze do not.
        weight_paths = logger.glob("**/models/**/f_lm.pkl")
        logger.print('found these experiments')
        logger.print(*weight_paths, sep="\n")
        for p in weight_paths:
            parameter_path = normpath(pJoin(dirname(p), '..', '..', 'parameters.pkl'))
            env_id, local_metric, latent_dim = \
                logger.get_parameters(
                    'Args.env_id', 'Args.local_metric', 'Args.latent_dim',
                    path=parameter_path, default=None)
            logger.abspath(p)

            kwargs.append(dict(env_id=env_id, load_local_metric=logger.abspath(p),
                               local_metric=local_metric, latent_dim=latent_dim))

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
