from plan2vec_experiments import instr
from plan2vec.plan2vec.plan2vec_rope import DEBUG, Args, main
import jaynes


def common_config():
    Args.num_epochs = 1000
    Args.lr = 1e-3
    Args.gamma = 0.97
    Args.visualization_interval = 10

    Args.term_r = 1.05
    Args.batch_n = 200
    Args.optim_epochs = 200
    Args.optim_batch_size = 64

    # load local metric function
    exp_path = "episodeyang/plan2vec/2019/04-22/rope_local_metric/14.40/50.184777"
    weight_file = "models/local_metric_093-031.pkl"
    Args.load_local_metric = f"/{exp_path}/{weight_file}"

    # use pre-computed pairwise matrix.
    Args.load_top_k = "/episodeyang/plan2vec/2019/04-24/rope_pairwise/run-21.27.44/top_24.pkl"


def plan2vec(**kwargs):
    common_config()
    Args.binary_reward = False
    Args.update(kwargs)
    _ = instr(main, __postfix="parameter-sweep-4/plan2vec", **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


def plan2vec_binary_reward(**kwargs):
    common_config()
    Args.binary_reward = True
    Args.update(kwargs)
    _ = instr(main, __postfix="parameter-sweep-4/plan2vec_binary_reward", **vars(Args), _DEBUG=vars(DEBUG))
    jaynes.run(_)


if __name__ == "__main__":
    # jaynes.config("devfair")
    # plan2vec()
    # #
    # jaynes.listen()

    # raise NotImplementedError('Need to change prefix from parameter-sweep-4 to something else. Was parameter-sweep before.')
    jaynes.config("vector-gpu")
    for Args.lr in [3e-3, 1e-3, 3e-4, 1e-4]:
        for Args.target_update in [0.9, 0.95, 0.97, 0.99]:
            plan2vec(lr=Args.lr, target_update=Args.target_update)
            plan2vec_binary_reward(lr=Args.lr, target_update=Args.target_update)

    jaynes.listen()
