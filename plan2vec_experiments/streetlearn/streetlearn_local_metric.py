from plan2vec.plan2vec.local_metric_streetlearn import main, Args

if __name__ == "__main__":
    from plan2vec_experiments import instr, config_charts, config_charts

    import jaynes

    jaynes.config()

    for Args.local_metric in ["LocalMetricConvDeep", "ResNet18L2"]:
        for Args.lr in [1e-5]:  # 1e-4, 3e-5, 6e-5,
            for seed in range(3):
                _ = instr(main,
                          seed=seed * 100, lr=Args.lr, num_epochs=50,
                          checkpoint_interval=100,
                          __postfix=f"manhattan-medium/{Args.local_metric}/lr-({Args.lr})", **vars(Args))
                config_charts()
                jaynes.run(_)

    jaynes.listen()
