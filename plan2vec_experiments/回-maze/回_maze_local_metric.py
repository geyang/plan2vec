if __name__ == "__main__":
    from plan2vec.plan2vec.local_metric_state import main
    from plan2vec_experiments import instr, config_charts

    from termcolor import cprint

    cprint('Training on cluster', 'green')
    import jaynes

    jaynes.config("vector")
    for seed in range(3):
        jaynes.run(instr(main, env_id="回MazeDiscreteIdLess-v0", seed=seed * 100, num_epochs=100, n_rollouts=2000))
        config_charts()

    jaynes.listen()
