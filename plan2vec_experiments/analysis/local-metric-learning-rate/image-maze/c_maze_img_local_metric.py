from plan2vec.plan2vec.local_metric_img import main

if __name__ == "__main__":
    from plan2vec_experiments import instr, config_charts

    if False:
        thunk(main, env_id="CMazeDiscreteImgIdLess-v0", seed=10 * 100, num_epochs=20)()
        config_charts()
    elif True:
        from termcolor import cprint

        cprint('Training on cluster', 'green')
        import jaynes

        jaynes.config("vector-gpu")

        for seed in range(25, 30):
            jaynes.run(instr(main, env_id="CMazeDiscreteImgIdLess-v0", seed=seed * 100, num_epochs=40))
            config_charts()

        jaynes.listen()
