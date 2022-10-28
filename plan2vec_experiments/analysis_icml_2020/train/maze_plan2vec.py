"""
charts:
  - {glob: "figures/**/embed*.png", type: file}
  - {xKey: epoch, yKey: "success_rate", yDomain: [0, 1]}
  - {glob: "figures/**/value_map*.png", type: file}
keys:
  - run.status
  - Args.lr
  - Args.binary_reward
  - Args.n_rollouts
  - {metrics: "success_rate/mean"}
  - Args.global_metric
"""
from plan2vec_experiments import instr, config_charts
from plan2vec.plan2vec.plan2vec_img import DEBUG, Args, main


def common_config():
    Args.num_epochs = 1000
    Args.lr = 3e-4
    Args.gamma = 1
    Args.target_update = 0.9
    Args.top_k = 30  # remove
    Args.H = 50
    Args.r_scale = 0.2

    Args.n_rollouts = 200

    Args.checkpoint_interval = 200

    # Args.optim_batch_size = 128
    # Args.optim_epochs = 50
    # Args.batch_n = 100

    Args.env_id = "CMazeDiscreteImgIdLess-v0"
    local_metric_exp_path = "episodeyang/plan2vec/2019/05-07/c-maze-image/c_maze_local_metric/21.19/10.332015"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


def plan2vec(**_):
    common_config()

    Args._update(**_)

    thunk = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    config_charts(__doc__)
    jaynes.run(thunk)


def plan2vec_binary_reward():
    common_config()
    Args.binary_reward = True

    thunk = instr(main, **vars(Args), _DEBUG=vars(DEBUG))
    config_charts(__doc__)
    jaynes.run(thunk)


if __name__ == "__main__":
    import jaynes

    # architecturs = ['ResNet18L2', 'ResNet18Kernel',
    #                 'ResNet18CoordAsymmetricL2', 'ResNet18AsymmetricL2']

    jaynes.config()

    plan2vec(global_metric='ResNet18CoordL2')

    jaynes.listen()
