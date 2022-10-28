import jaynes
from params_proto.neo_hyper import Sweep
from plan2vec_experiments import instr, config_charts
from plan2vec.plan2vec.maze_uvpn_image import Args, train_advantage, eval_advantage, eval_planning_policy, \
    train_local_metric


def advantage():
    jaynes.config('local')
    thunk = instr(eval_advantage,
                  env_id="CMazeDiscreteImgIdLess-v0",
                  weight_path='/amyzhang/plan2vec/2020/02-05/neo_plan2vec/uvpn_image/maze_uvpn_image/'
                              'train-inverse-model-small/11.06/lr-6e-05-seed-400/03.638480/models/0095/adv.pkl')
    jaynes.run(thunk)


def train_advantage_():
    from plan2vec_experiments import instr, config_charts

    with Sweep(Args) as sweep:
        Args.num_rollouts = 1000
        Args.num_epochs = 100
        Args.checkpoint_interval = 5
        with sweep.zip:
            Args.adv_lr = [3e-4, 1e-4, 6e-5, 3e-5,
                           3e-5, 1e-5, 1e-5, 1e-5, ]
            Args.seed = [200, 300, 400, 500, 600, 700, 800, 900, ]

    for deps in sweep:
        thunk = instr(train_advantage,
                      env_id="CMazeDiscreteImgIdLess-v0",
                      # load_adv="/amyzhang/plan2vec/2020/01-28/neo_plan2vec/tweaking_maze_adv/"
                      #          "maze_load_adv_tweak/longer/2xv_s_s/07.44/CMazeDiscreteImgIdLess-v0/"
                      #          "ams-0.1/hard/34.121645/models/3800/adv.pkl",
                      __prefix="train-inverse-model-small",
                      __postfix=f"lr-{Args.adv_lr}-seed-{Args.seed}"
                      )
        jaynes.run(thunk, deps=deps)
        config_charts("""
            charts:
            - yKey: loss/mean
              xKey: epoch
            - yKey: accuracy/mean
              xKey: epoch
            - yKey: train/accuracy/mean
              xKey: epoch
            keys:
            - run.status
            - Args.adv_lr
            - Args.n_rollouts
            """)


def planning_policy():
    from params_proto.neo_hyper import Sweep
    from ml_logger import logger

    with Sweep(Args) as sweep:
        # Args.env_id = "CMazeDiscreteImgIdLess-v0"

        Args.sample_env_id = "CMazeDiscreteImgIdLess-v0"

        Args.env_pos = [-0.15, 0.15]
        Args.env_goal = [-0.15, -0.15]

        Args.dump_plans = True

        Args.neighbor_r = 1.
        Args.neighbor_r_min = 0.5
        Args.prune_r = 0.2
        Args.latent_dim = 10
        Args.num_rollouts = 400
        Args.search_alg = "dijkstra"
        # Args.local_metric = "LocalMetricConv"
        # Args.load_local_metric = "/geyang/plan2vec/2020/02-04/neo_plan2vec/uvpn_image/pretrains/" \
        #                          "local_metric_timing/pretrain-local-metric/10.33/LocalMetricConv/" \
        #                          "36.026403/models/0100/f_lm.pkl"

        Args.local_metric = "LocalMetricConvLarge"
        Args.load_local_metric = "/geyang/plan2vec/2019/12-16/analysis/local-metric-analysis/all_local" \
                                 "_metric/11.55/img_maze/CMazeDiscreteImgIdLess-v0/22.120194/models/" \
                                 "local_metric.pkl"
        Args.load_adv = '/amyzhang/plan2vec/2020/02-05/neo_plan2vec/uvpn_image/maze_uvpn_image/' \
                        'train-inverse-model-small/11.06/lr-6e-05-seed-400/03.638480/models/0095/adv.pkl'

        Args.num_evals = 2
        Args.num_eval_rollouts = 1
        Args.eval_limit = 1000
        Args.seed = 200
        with sweep.zip:
            Args.eval_limit = [50, 1000]
            Args.env_id = ["CMazeDiscreteImgIdLess-v0", "FourRoomDiscreteImgIdLess-v0"]
        #     Args.seed = [200, 300, 400, 500, 600, 700]

    for deps in sweep[1:]:
        # thunk = instr(train, deps, __prefix="pretrain-local-metric", __postfix=f"{Args.local_metric}")
        thunk = instr(eval_planning_policy, deps, __prefix="adaptation",
                      __postfix=f"{Args.local_metric}-r-{Args.neighbor_r}-rmin-{Args.neighbor_r_min}/{Args.env_id}")
        jaynes.run(thunk, )
        config_charts("""
            charts:
            - yKey: success/mean
              xKey: eval_step
            keys:
            - run.status
            - Args.prune_r
            - Args.neighbor_r
            - Args.neighbor_r_min
            """)
        logger.log_text(__doc__, "README.md")


def learning_f_local():
    from params_proto.neo_hyper import Sweep
    from ml_logger import logger

    with Sweep(Args) as sweep:
        # Args.env_id = "CMazeDiscreteImgIdLess-v0"

        Args.sample_env_id = "CMazeDiscreteImgIdLess-v0"

        Args.env_pos = [-0.15, 0.15]
        Args.env_goal = [-0.15, -0.15]

        Args.dump_plans = True

        Args.neighbor_r = 1.
        Args.neighbor_r_min = 0.5
        Args.prune_r = 0.2
        Args.latent_dim = 10
        Args.num_rollouts = 400
        Args.search_alg = "dijkstra"
        # Args.local_metric = "LocalMetricConv"
        # Args.load_local_metric = "/geyang/plan2vec/2020/02-04/neo_plan2vec/uvpn_image/pretrains/" \
        #                          "local_metric_timing/pretrain-local-metric/10.33/LocalMetricConv/" \
        #                          "36.026403/models/0100/f_lm.pkl"

        Args.local_metric = "LocalMetricConvLarge"
        Args.load_local_metric = "/geyang/plan2vec/2019/12-16/analysis/local-metric-analysis/all_local" \
                                 "_metric/11.55/img_maze/CMazeDiscreteImgIdLess-v0/22.120194/models/" \
                                 "local_metric.pkl"
        Args.load_adv = '/amyzhang/plan2vec/2020/02-05/neo_plan2vec/uvpn_image/maze_uvpn_image/' \
                        'train-inverse-model-small/11.06/lr-6e-05-seed-400/03.638480/models/0095/adv.pkl'

        Args.num_evals = 2
        Args.num_eval_rollouts = 1
        Args.eval_limit = 1000
        Args.seed = 200
        with sweep.zip:
            Args.eval_limit = [50, 1000]
            Args.env_id = ["CMazeDiscreteImgIdLess-v0", "FourRoomDiscreteImgIdLess-v0"]
        #     Args.seed = [200, 300, 400, 500, 600, 700]

    for deps in sweep[1:]:
        # thunk = instr(train, deps, __prefix="pretrain-local-metric", __postfix=f"{Args.local_metric}")
        thunk = instr(train_local_metric, deps, __prefix="local_metric",
                      __postfix=f"{Args.local_metric}/{Args.env_id}")
        jaynes.run(thunk, )
        config_charts("""
            charts:
            - yKey: loss/mean
              xKey:  epoch
            - yKey: d_bar/mean
              xKey:  epoch
            - yKey: d_null/mean
              xKey:  epoch
            - yKey: d_shuffle/mean
              xKey:  epoch
            keys:
            - run.status
            - Args.prune_r
            - Args.neighbor_r
            - Args.neighbor_r_min
            """)
        logger.log_text(__doc__, "README.md")


if __name__ == '__main__':
    # jaynes.config('local')
    # train_advantage_()
    # planning_policy()
    learning_f_local()
    jaynes.listen()
