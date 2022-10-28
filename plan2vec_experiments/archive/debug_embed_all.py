"""
debug script for the embedding function. Can delete afterward.
"""
import jaynes
from plan2vec.dqn.point_mass_q_learning import Args, train
from plan2vec_experiments import instr
from ml_logger import logger
from os.path import join as p_join

if __name__ == "__main__":
    ts = logger.now('%H.%M.%S')


    def run_local_stuff(job_postfix="", index=0, partition=None):
        prefix = p_join(logger.stem(__file__), ts, job_postfix + logger.now('%f'))
        jaynes.config('local')
        _ = instr(train, prefix)
        jaynes.run(_, **vars(Args))
        import time
        time.sleep(5)


    Args.matplotlib_backend = "Agg"
    Args.num_envs = 3
    Args.eps_greedy = 0.8
    Args.num_episodes = 20
    Args.visualize_interval = 10

    with logger.SyncContext():
        Args.env_id = "GoalMassDiscreteIdLess-v0"
        # Args.q_fn = "vanilla"
        # Args.q_fn = "shared-encoder"
        # Args.q_fn = "l2-embed-T"
        # Args.forward_coef = 1
        # Args.q_fn = "l2-embed"
        # run_local_stuff(Args.q_fn + "-", partition='dev')

        Args.learning_mode = "passive"
        Args.q_fn = "l2-embed"
        run_local_stuff(Args.q_fn + "-", partition='dev')

        Args.env_id = "GoalMassDiscreteImgIdLess-v0"
        Args.obs_key = 'img'
        Args.goal_key = 'goal_img'

        # Args.q_fn = "vanilla-conv"
        # Args.q_fn = "shared-encoder-conv"
        # Args.q_fn = "l2-embed-T-conv"
        # Args.forward_coef = 1
        Args.q_fn = "l2-embed-conv"
        run_local_stuff(Args.q_fn + "-", partition='dev')
