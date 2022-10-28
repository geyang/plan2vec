from plan2vec.plan2vec.local_metric_state import main, Args
from plan2vec_experiments import instr, config_charts

if __name__ == "__main__":
    from termcolor import cprint

    cprint('Training on cluster', 'green')
    import jaynes

    jaynes.config("vector")

    for seed in range(3):
        jaynes.run(instr(main, env_id="CMazeDiscreteIdLess-v0", seed=seed * 100, num_epochs=200))
        config_charts()

    jaynes.listen()
