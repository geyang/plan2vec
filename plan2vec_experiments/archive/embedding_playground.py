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
    Args.num_envs = 3
    # for Args.lr in [1e-4, 3e-4, 1e-3]:
    #     for Args.weight_decay in [1e-6, 1e-5, 1e-4]:
    # for Args.q_fn in ["shared-encoder", "l2-embed", "l2-embed-T", 'vanilla']:
    for Args.env_id in ["GoalMassDiscreteIdLess-v0", "GoalMassDiscreteIdLessTerm-v0"]:
        for Args.q_fn in ["shared-encoder", "l2-embed", 'l2-embed-T', 'vanilla']:
            Args.num_episodes = 3000 if Args.q_fn == "vanilla" else 10000
            learning_rates = [0.001] if Args.q_fn == "vanilla" else [0.0001, 0.0003, 0.001]
            for Args.lr in learning_rates:
                if Args.q_fn == "l2-embed":
                    for Args.embed_p in [1, 1.1, 1.5, 1.9, 2]:
                        run_stuff(f"{Args.env_id}-{Args.q_fn}-lr({Args.lr})-p({Args.embed_p})-", partition="vector")
                else:
                    run_stuff(f"{Args.env_id}-{Args.q_fn}-lr({Args.lr})-", partition="vector")

    jaynes.listen(600)
