from plan2vec.plotting.maze_world.connect_the_dots_image_maze import main

if __name__ == "__main__":
    from plan2vec_experiments import instr
    import jaynes

    jaynes.config('vector-gpu')

    for env_id, metric_path in zip([
        'GoalMassDiscreteImgIdLess-v0',
        'CMazeDiscreteImgIdLess-v0',
        '回MazeDiscreteImgIdLess-v0'
    ], [
        "episodeyang/plan2vec/2019/05-10/goal-mass-image/local_metric/14.15/38.768056",
        # "episodeyang/plan2vec/2019/05-07/c-maze-image/c_maze_local_metric/21.19/10.332015",
        # "episodeyang/plan2vec/2019/05-20/c-maze-image/c_maze_img_local_metric/16.01/50.590108",
        "episodeyang/plan2vec/2019/05-20/c-maze-image/c_maze_img_local_metric/23.57/27.708595",
        "episodeyang/plan2vec/2019/05-10/回-maze-image/回_maze_img_local_metric/13.28/59.358543"
    ]):
        jaynes.run(instr(main,
                         env_id=env_id,
                         n_rollouts=100,
                         # n_rollouts=140 if env_id.startswith("Goal") else 120,
                         load_local_metric=f"/{metric_path}/models/local_metric.pkl"))

    jaynes.listen()
