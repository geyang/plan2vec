import jaynes
from plan2vec.dqn.point_mass_q_learning import Args, train
from plan2vec_experiments import instr
from ml_logger import logger
from os.path import join as p_join

if __name__ == "__main__":
    ts = logger.now('%H.%M.%S')


    def run_stuff(job_postfix="", index=0, partition=None):
        prefix = p_join(logger.stem(__file__), ts, job_postfix + logger.now('%f'))
        jaynes.config(runner=dict(partition=partition or ('vector' if index < 20 else "vector"), name=prefix),
                      display=logger.now("%f"), n_cpu=7)
        _ = instr(train, prefix)
        jaynes.run(_, **vars(Args))
        import time
        time.sleep(5)


    Args.matplotlib_backend = "Agg"
    Args.env_id = "GoalMassDiscreteIdLess-v0"
    Args.num_envs = 3
    Args.num_episodes = 3000
    # for Args.lr in [1e-4, 3e-4, 1e-3]:
    #     for Args.weight_decay in [1e-6, 1e-5, 1e-4]:
    #         Args.q_fn = "vanilla"
    #         run_stuff(f"{Args.env_id}-{Args.q_fn}-lr({Args.lr})-wd({Args.weight_decay})", partition="vector")
    #
    # jaynes.listen(600)

    # Args.visualize_interval = 10

    # jaynes.config(mode="local")
    # # Args.q_fn = 'l2-embed'
    # # Args.q_fn = 'shared-encoder'
    # Args.q_fn = 'vanilla'
    # thunk(train, logger.stem(__file__), ts, logger.now('%f'))()
    # exit()

    Args.q_fn = "vanilla"
    # run_stuff(Args.q_fn + "-", partition='dev')
    # jaynes.listen(600)
    # exit()

    i = 0
    Args.lr = 1e-3
    Args.wd = 1e-6
    Args.target_update = 3
    for Args.env_id in ["GoalMassDiscreteIdLess-v0"]:
        for Args.start_seed in range(5):
            i += 1
            Args.q_fn = "vanilla"
            run_stuff(f"{Args.env_id}-{Args.q_fn}-seed{Args.start_seed}", i)

            i += 1
            Args.q_fn = "shared-encoder"
            run_stuff(f"{Args.env_id}-{Args.q_fn}-seed{Args.start_seed}", i)

            i += 1
            Args.embed_id_init = False
            Args.embed_id_init_all = False
            Args.lr = 3e-4
            Args.q_fn = 'l2-embed-T'
            run_stuff(f"{Args.env_id}-{Args.q_fn}-seed{Args.start_seed}", i)

            i += 1
            Args.q_fn = "l2-embed"
            Args.env_id = "GoalMassDiscrete-v0"
            run_stuff(f"{Args.env_id}-{Args.q_fn}-seed{Args.start_seed}", i)

    jaynes.listen(600)
