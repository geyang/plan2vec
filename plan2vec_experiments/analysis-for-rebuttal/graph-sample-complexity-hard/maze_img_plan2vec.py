from plan2vec_experiments import instr, config_charts
from plan2vec.plan2vec.plan2vec_img import DEBUG, Args, main
import jaynes
from params_proto.neo_hyper import Sweep

ENVS = [
    # "GoalMassDiscreteImgIdLess-v0",
    "回MazeDiscreteImgIdLess-v0",
    "CMazeDiscreteImgIdLess-v0",
]
LOCAL_METRIC_PATHS = [
    f"{key}/models/local_metric.pkl" for key in [
        # "/geyang/plan2vec/2019/06-21/goal-mass-image/goal_mass_img_local_metric/22.39/21.462733",
        "/geyang/plan2vec/2019/06-21/回-maze-image/回_maze_img_local_metric/22.39/24.875458",
        "/geyang/plan2vec/2019/06-21/c-maze-image/c_maze_img_local_metric/22.39/28.297534"
    ]
]
SEEDS = [s * 100 for s in range(10, 15)]#[:2]


def plan2vec():
    Args.n_rollouts = 800
    Args.timesteps = 2
    Args.latent_dim = 3
    Args.term_r = 0.04
    Args.num_epochs = 200
    Args.batch_n = 20
    Args.H = 30
    Args.gamma = 0.97
    Args.r_scale = 0.2
    Args.optim_epochs = 32
    Args.optim_batch_size = 128
    Args.lr = 0.0003
    Args.checkpoint_interval = 40
    Args.global_metric = "ResNet18L2"

    with Sweep(Args) as sweep:
        with sweep.product:
            # Args.optim_batch_size = [16, 32, 128]

            with sweep.zip:
                Args.env_id = ENVS
                Args.load_local_metric = LOCAL_METRIC_PATHS

            Args.start_seed = SEEDS

            # Args.n_rollouts = [10, 20, 30, 50, 100, 200, 300, 500, 800, ]
            # Args.num_envs = [1, 10, 10, 10, 20, 20, 20, 20, 20, ]

    for deps in sweep:
        _ = instr(main, deps, __postfix=f"{Args.env_id}/{Args.optim_batch_size}")
        config_charts("""
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
        """)
        jaynes.run(_)


def deplete_12():
    from ml_logger import logger
    if "gpu012" in logger.hostname:
        import time
        print(logger.hostname, "host is defective. We are blocking this run.")
        time.sleep(300)
        print('blocking is finished.')
    else:
        print(logger.hostname, "host is not defective, releasing right away.")


if __name__ == "__main__":
    jaynes.config()

    for i in range(10):
        jaynes.run(deplete_12)

    plan2vec()

    jaynes.listen(600)
