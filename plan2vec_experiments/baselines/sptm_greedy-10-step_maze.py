from plan2vec.plan2vec.plan2vec_img import main, Args

if __name__ == "__main__":
    import jaynes
    from plan2vec_experiments import instr

    Args.load_local_metric = "same as before"
    Args.load_global_metric = "same as local metric"

    jaynes.config("vector-gpu")

    for env_id, exp_path in {
        "GoalMassDiscreteImgIdLess-v0":
            "/geyang/plan2vec/2019/06-21/goal-mass-image/goal_mass_img_local_metric/22.39/21.462733",
        "回MazeDiscreteImgIdLess-v0":
            "/geyang/plan2vec/2019/06-21/回-maze-image/回_maze_img_local_metric/22.39/24.875458",
        "CMazeDiscreteImgIdLess-v0":
            "/geyang/plan2vec/2019/06-21/c-maze-image/c_maze_img_local_metric/22.39/28.297534"
    }.items():
        local_metric_path = f"{exp_path}/models/local_metric.pkl"

        _ = instr(main,
                  evaluate=True,
                  lr=0,
                  env_id=env_id,
                  plan_steps=10,
                  visualization_interval=False,
                  global_metric="LocalMetricConvLarge",
                  latent_dim=32,  # use the same as local metric
                  load_local_metric=local_metric_path,
                  load_global_metric=local_metric_path
                  )

        jaynes.run(_)

    jaynes.listen()
