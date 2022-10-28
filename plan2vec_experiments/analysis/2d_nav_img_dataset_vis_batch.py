from plan2vec.plotting.maze_world.connect_the_dots_image_maze import Args, main

if __name__ == "__main__":
    import jaynes
    from plan2vec_experiments import instr, RUN
    from os.path import join as pJoin
    from ml_logger import logger

    jaynes.config('vector-gpu')

    env_id = 'CMazeDiscreteImgIdLess-v0'

    logger.configure(log_directory=RUN.server, register_experiment=False)
    prefix = "/episodeyang/plan2vec/2019/05-20/c-maze-image/c_maze_img_local_metric/23.57"
    weight_paths = logger.glob("**/models/local_metric.pkl", prefix)

    for path in weight_paths:
        jaynes.run(instr(main,
                         env_id=env_id,
                         n_rollouts=100,
                         load_local_metric=pJoin(prefix, path)
                         ))

    jaynes.listen()
