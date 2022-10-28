from termcolor import cprint

from plan2vec.plan2vec.local_metric_state import main
from plan2vec_experiments import instr, config_charts

if __name__ == "__main__":
    cprint('Training on cluster', 'green')
    import jaynes

    jaynes.config("vector")

    for i in range(3):
        jaynes.run(instr(main, seed=i * 100, num_epochs=200, n_rollouts=2000))
        config_charts()

    jaynes.listen()
