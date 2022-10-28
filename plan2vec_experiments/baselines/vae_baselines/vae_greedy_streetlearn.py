from plan2vec.plan2vec.plan2vec_streetlearn_2 import main, Args

if __name__ == "__main__":
    import jaynes
    from plan2vec_experiments import instr, config_charts

    Args.load_local_metric = "same as before"
    Args.load_global_metric = "same as local metric"

    jaynes.config("vector-gpu")

    Args.visualization_interval = False

    local_metric_exp_path = "/geyang/plan2vec/2019/07-31/streetlearn/local_metric/manhattan-xl/LocalMetricConvDeep/lr-(3e-05)/00.37/13.921429"

    for env_id, global_checkpoint in [
        ["manhattan-tiny",
         "/geyang/plan2vec/2019/07-31/baselines/vae/streetlearn_vae/dim-(2)/manhattan-tiny/lr-(3e-05)/06.57/11.442504"],
        ["manhattan-small",
         "/geyang/plan2vec/2019/07-31/baselines/vae/streetlearn_vae/dim-(2)/manhattan-small/lr-(3e-05)/06.57/23.367555"],
        ["manhattan-medium",
         "/geyang/plan2vec/2019/07-31/baselines/vae/streetlearn_vae/dim-(2)/manhattan-medium/lr-(3e-05)/06.57/30.313456"],
        ["manhattan-large",
         "/geyang/plan2vec/2019/07-31/baselines/vae/streetlearn_vae/dim-(2)/manhattan-large/lr-(3e-05)/06.57/37.217606"]
    ]:
        latent_dim = 2
        _ = instr(main,
                  __postfix=f"{env_id}",
                  lr=0,
                  env_id=env_id,
                  data_path=f"~/fair/streetlearn/processed-data/{env_id}",
                  latent_dim=latent_dim,
                  plan_steps=1,
                  neighbor_r=0.9,
                  evaluate=True, term_r=1.2e-4 * float(2),
                  num_epochs=200,
                  visualization_interval=False,
                  global_metric="ResNet18L2",
                  load_local_metric=f"{local_metric_exp_path}/models/local_metric_400.pkl",
                  load_global_metric=f"{global_checkpoint}/checkpoints/encoder.pkl",
                  load_global_metric_matcher=lambda d, k, *_: d[k.replace('embed.', '')]
                  )

        config_charts(path="streetlearn.charts.yml")
        jaynes.run(_)

    jaynes.listen()
