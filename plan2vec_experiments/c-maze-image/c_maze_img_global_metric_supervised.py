from plan2vec.plan2vec.global_metric_img import main

if __name__ == "__main__":
    from plan2vec_experiments import instr

    if False:
        _ = thunk(main,
                  env_id="CMazeDiscreteImgIdLess-v0",
                  seed=10 * 100, num_epochs=20)
        _()
    elif True:
        from termcolor import cprint

        cprint('Training on cluster', 'green')
        import jaynes

        jaynes.config("vector-gpu")

        for lr in [3e-5, 1e-4, 3e-4]:
            for seed in range(25, 30):
                _ = instr(main, env_id="CMazeDiscreteImgIdLess-v0", seed=seed * 100, lr=lr, num_epochs=40)
                jaynes.run(_)

        jaynes.listen()
