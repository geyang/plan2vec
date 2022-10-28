import jaynes
from plan2vec.dqn.point_mass_q_learning import Args, train
from plan2vec_experiments import instr
from ml_logger import logger
from os.path import join as p_join

if __name__ == "__main__":
    ts = logger.now('%H.%M.%S')


    def run_stuff(job_postfix="", index=0):
        prefix = p_join(logger.stem(__file__), ts, job_postfix + logger.now('%f'))
        jaynes.config(runner=dict(partition='vector' if index < 20 else "vector", name=prefix),
                      display=logger.now("%f"))
        _ = instr(train, prefix)
        jaynes.run(_, **vars(Args))
        import time
        time.sleep(5)


    Args.matplotlib_backend = "Agg"
    Args.env_id = "PointMassDiscrete-v0"
    Args.num_episodes = 3000
    # Args.visualize_interval = 10

    # jaynes.config(mode="local")
    # # Args.q_fn = 'l2-embed-T'
    # Args.q_fn = 'shared-encoder'
    # thunk(train, logger.stem(__file__), ts, logger.now('%f'))()
    # exit()

    i = 0

    for Args.env_id in ["PointMassDiscreteIdLess-v0", "PointMassDiscrete-v0"]:
        for Args.target_update in [0.1, 10]:
            i += 1
            Args.q_fn = "vanilla"
            run_stuff(Args.q_fn + "-", i)

            i += 1
            Args.q_fn = "shared-encoder"
            run_stuff(Args.q_fn + '-', i)

            i += 1
            Args.embed_id_init = False
            Args.embed_id_init_all = False
            Args.q_fn = "l2-embed"
            run_stuff(Args.q_fn + "-", i)

            i += 1
            Args.q_fn = 'l2-embed-T'
            run_stuff(Args.q_fn + "-", i)

    jaynes.listen(600)
