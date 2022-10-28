if __name__ == "__main__":
    from plan2vec.plan2vec.local_metric_rope import main, Args
    from plan2vec_experiments import instr

    import jaynes
    from termcolor import cprint

    cprint('Training on cluster', 'green')
    jaynes.config()

    Args.num_epochs = 100
    for seed in range(3):
        jaynes.run(instr(main, seed=seed, lr=Args.lr))

    jaynes.listen()
