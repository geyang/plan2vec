from plan2vec_experiments import instr
from plan2vec.plan2vec.plan2vec_img import DEBUG, Args, main


def common_config():
    Args.start_seed = 500

    Args.n_rollouts = 100

    Args.num_epochs = 5000
    Args.lr = 3e-5
    Args.gamma = 0.97
    Args.target_update = 0.9
    Args.H = 20
    Args.r_scale = 0.2

    Args.optim_batch_size = 128
    Args.optim_epochs = 50
    Args.batch_n = 100

    local_metric_exp_path = "episodeyang/plan2vec/2019/04-30/goal-mass-image/local_metric/13.04/50.045261"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


def pairwise_map_reduce(n=10):
    from termcolor import cprint
    from tqdm import trange
    import jaynes

    cprint('Computing pairwise with map-reduce', 'green')

    common_config()

    jaynes.config("vector-gpu")

    _ = instr(main, **vars(Args))
    for k in trange(n, desc="launching map-reduce jobs"):
        jaynes.run(_, map_reduce=dict(k=k, n=n))
    jaynes.listen()


def pairwise_debug():
    from termcolor import cprint
    import jaynes

    cprint('Computing pairwise with map-reduce', 'green')

    common_config()

    jaynes.config("vector-gpu")

    _ = instr(main, **vars(Args))
    jaynes.run(_, map_reduce=dict(k=5, n=10))
    jaynes.listen()


if __name__ == "__main__":
    # pairwise_debug()
    pairwise_map_reduce(n=10)
