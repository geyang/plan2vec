"""
This is just to show that the vision based local-metric does not generalize across
different task domains (except 回-Maze -> Goal-maze). The black wall in the middle
affects the normalization of the CNN.
"""
from plan2vec.plotting.maze_world.connect_the_dots_image_maze import Args, main

if __name__ == "__main__":
    import jaynes
    from plan2vec_experiments import instr, RUN
    from os.path import join as pJoin
    from ml_logger import logger

    jaynes.config('vector-gpu')

    # Args.latent_dim = 50

    logger.configure(log_directory=RUN.server, register_experiment=False)
    prefix = "/episodeyang/plan2vec/2019/05-10/回-maze-image/回_maze_img_local_metric"
    path = "13.28/59.358543/models/local_metric.pkl"
    jaynes.run(instr(main,
                     env_id='回MazeDiscreteImgIdLess-v0',
                     n_rollouts=100, load_local_metric=pJoin(prefix, path)))
    jaynes.run(instr(main,
                     n_rollouts=100, load_local_metric=pJoin(prefix, path)))
    jaynes.listen()
