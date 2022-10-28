from plan2vec.schedules import LinearAnneal, CosineAnneal
from plan2vec_experiments import instr

if __name__ == "__main__":
    from plan2vec.dqn.point_mass_q_learning import Args, train
    from ml_logger import logger

    Args.env_id = 'CMazeDiscreteIdLess-v0'
    Args.num_episodes = 10000
    Args.num_envs = 3
    Args.latent_dim = 10
    Args.replay_memory = 10000
    Args.metric_summary_interval = 10
    Args.record_video_interval = None
    Args.wd = 1e-6
    Args.prioritized_replay = True
    Args.beta = LinearAnneal(0.4, min=1.0, n=Args.num_episodes + 1)
    Args.eps_greedy = 0.1  # CosineAnneal(0.5, min=0.05, n=Args.num_episodes + 1)

    Args.good_relabel = True

    time = logger.now('%H.%M.%S')

    params = []
    # Low-Dim
    for Args.q_fn in ['l2-embed-T', 'l2-embed', 'shared-encoder', 'vanilla']:
        Args.gamma = 1 if Args.q_fn.startswith("l2") else 0.98
        Args.lr = 3e-4 if Args.q_fn.startswith("l2") else 1e-3
        Args.forward_coef = 1. if Args.q_fn == 'l2-embed-T' else 0

        prefix = f"{logger.stem(__file__)}/{time}/{Args.env_id}-(fixed)-{Args.q_fn}"

        params.append(dict(prefix=prefix, Args=vars(Args)))
        # jaynes.config(runner=dict(partition="vector", name=prefix, n_cpu=10, n_gpu=1),
        #               display=logger.now("%f"))

    with logger.AsyncContext():
        p = params[3]
        _ = instr(train, p['prefix'])
        _(**p['Args'])
