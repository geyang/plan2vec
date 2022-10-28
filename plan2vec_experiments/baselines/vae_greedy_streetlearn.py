from plan2vec.plan2vec.plan2vec_streetlearn_2 import main, Args

if __name__ == "__main__":
    import jaynes
    from plan2vec_experiments import instr, config_charts

    Args.load_local_metric = "same as before"
    Args.load_global_metric = "same as local metric"

    jaynes.config("vector-gpu")

    Args.visualization_interval = False

    local_metric_exp_path = "/geyang/plan2vec/2019/07-31/streetlearn/local_metric/manhattan-xl/LocalMetricConvDeep/lr-(3e-05)/00.37/13.921429"

    for env_id, epx_path in {
        "manhattan-tiny":
        # "/geyang/plan2vec/2019/07-30/vae/streetlearn_vae/dim-(3)/manhattan-tiny/lr-(0.003)/23.00/08.673423"
            "/geyang/plan2vec/2019/07-30/vae/streetlearn_vae-incomplete/dim-(2)/manhattan-tiny/lr-(0.0003)/22.59/48.194736",
        "manhattan-small":
        # "/geyang/plan2vec/2019/07-30/vae/streetlearn_vae/dim-(3)/manhattan-small/lr-(0.003)/23.00/14.916093",
            "/geyang/plan2vec/2019/07-30/vae/streetlearn_vae-incomplete/dim-(2)/manhattan-small/lr-(3e-05)/22.59/51.306878",
        "manhattan-medium": "",
        # "manhattan-large": ""
    }.items():
        _ = instr(main,
                  __postfix=f"{env_id}",
                  lr=0,
                  env_id=env_id,
                  data_path=f"~/fair/streetlearn/processed-data/{env_id}",
                  plan_steps=1,
                  neighbor_r=0.9,
                  evaluate=True, term_r=1.2e-4 * float(2),
                  num_epochs=200,
                  visualization_interval=False,
                  global_metric="LocalMetricConvDeep",
                  latent_dim=32,  # use the same as local metric
                  load_local_metric=f"{local_metric_exp_path}/models/local_metric_400.pkl",
                  load_global_metric=f"{local_metric_exp_path}/models/local_metric_400.pkl")

        config_charts(path="streetlearn.charts.yml")
        jaynes.run(_)
        jaynes.listen()

    jaynes.listen()
