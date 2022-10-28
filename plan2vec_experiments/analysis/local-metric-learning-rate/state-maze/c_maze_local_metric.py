from plan2vec.plan2vec.local_metric_state import main, Args
from plan2vec_experiments import instr, config_charts

if __name__ == "__main__":

    if False:
        _ = thunk(main, env_id="CMazeDiscreteIdLess-v0", seed=10 * 100, num_epochs=100)()
    elif True:
        from termcolor import cprint

        cprint('Training on cluster', 'green')
        import jaynes

        jaynes.config("vector")

        for lr in [1e-4, 3e-4, 1e-3]:
            for seed in range(3):
                _ = instr(main, lr=lr, env_id="CMazeDiscreteIdLess-v0", seed=seed * 100, num_epochs=200)
                config_charts()
                jaynes.run(_)

        jaynes.listen()
