from plan2vec.plan2vec.plan2vec_img import main, Args

if __name__ == "__main__":
    import jaynes
    from plan2vec_experiments import instr, config_charts

    Args.load_local_metric = "same as before"
    Args.load_global_metric = "same as local metric"

    jaynes.config("vector-gpu")

    for env_id, exp_paths in {
        "GoalMassDiscreteImgIdLess-v0":
            ["/geyang/plan2vec/2019/06-21/goal-mass-image/goal_mass_img_local_metric/22.39/21.462733",
             "/geyang/plan2vec/2019/07-31/baselines/vae/maze_vae/dim-(2)/GoalMassDiscreteImgIdLess-v0/lr-(0.001)/20.35/21.561338"
             ],
        "回MazeDiscreteImgIdLess-v0":
            ["/geyang/plan2vec/2019/06-21/回-maze-image/回_maze_img_local_metric/22.39/24.875458",
             "/geyang/plan2vec/2019/07-31/baselines/vae/maze_vae/dim-(2)/回MazeDiscreteImgIdLess-v0/lr-(0.0003)/20.35/26.771705"
             ],
        "CMazeDiscreteImgIdLess-v0":
            ["/geyang/plan2vec/2019/06-21/c-maze-image/c_maze_img_local_metric/22.39/28.297534",
             "/geyang/plan2vec/2019/07-31/baselines/vae/maze_vae/dim-(2)/CMazeDiscreteImgIdLess-v0/lr-(0.0001)/20.35/32.059658"
             ]
    }.items():
        local_metric_path = f"{exp_paths[0]}/models/local_metric.pkl"
        vae_global_embed_path = f"{exp_paths[1]}/checkpoints/encoder.pkl"

        latent_dim = 2
        _ = instr(main,
                  evaluate=True,
                  lr=0,
                  env_id=env_id,
                  latent_dim=latent_dim,
                  plan_steps=1,
                  neighbor_r=0.9,
                  term_r=0.02,
                  num_epochs=200,
                  visualization_interval=False,
                  global_metric="ResNet18L2",
                  load_local_metric=local_metric_path,
                  load_global_metric=vae_global_embed_path,
                  load_global_metric_matcher=lambda d, k, *_: d[k.replace('embed.', '')]
                  )
        config_charts()
        jaynes.run(_)

    jaynes.listen()
