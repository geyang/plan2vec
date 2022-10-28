import jaynes
from params_proto.neo_hyper import Sweep

from plan2vec.plan2vec.maze_uvpn_image import Args, eval_planning_policy, \
    train_local_metric
from plan2vec_experiments import instr, config_charts


def learning_f_local():
    from params_proto.neo_hyper import Sweep
    from ml_logger import logger

    with Sweep(Args) as sweep:
        # Args.env_id = "CMazeDiscreteImgIdLess-v0"
        Args.env_id = "CMazeDiscreteImgIdLess-v0"

        Args.num_epochs = 100
        Args.checkpoint_interval = 100
        Args.checkpoint_overwrite = True
        Args.checkpoint_after = 50
        with sweep.product:
            Args.lr = [1e-3, 6e-4, 3e-4, 1e-4, 6e-5]
            Args.local_metric = ['LocalMetricConvLargeL2']
            Args.seed = [200]  # , 300, 400, 500, 600, 700]

    for deps in sweep:
        thunk = instr(train_local_metric, deps, __prefix="local_metric/hige_loss/lr-sweep",
                      __postfix=f"{Args.local_metric}-{Args.lr}/{Args.env_id}")
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


# if __name__ == '__main__':
#     jaynes.config()
#     learning_f_local()
#     jaynes.listen()


def planning_policy():
    from params_proto.neo_hyper import Sweep
    from ml_logger import logger

    with Sweep(Args) as sweep:
        Args.env_id = "CMazeDiscreteImgIdLess-v0"

        Args.env_pos = [-0.15, 0.15]
        Args.env_goal = [-0.15, -0.15]

        # Args.dump_plans = True

        Args.neighbor_r = 1.4
        Args.neighbor_r_min = 0.7
        Args.prune_r = 0.2
        Args.latent_dim = 10
        Args.num_rollouts = 800
        Args.search_alg = "dijkstra"

        Args.local_metric = "LocalMetricConvLargeL2"
        Args.load_local_metric = "/geyang/plan2vec/2020/02-08/neo_plan2vec/" \
                                 "uvpn_image/quick_eval_new_local_metric/" \
                                 "local_metric/hige_loss/lr-sweep/12.24/" \
                                 "LocalMetricConvLargeL2-6e-05/" \
                                 "CMazeDiscreteImgIdLess-v0/12.783076/models/0100/f_lm.pkl"

        Args.load_adv = '/amyzhang/plan2vec/2020/02-05/neo_plan2vec/uvpn_image/' \
                        'maze_uvpn_image/train-inverse-model-small/11.06/lr-6e-' \
                        '05-seed-400/03.638480/models/0095/adv.pkl'
        Args.num_evals = 4
        Args.num_eval_rollouts = 4
        Args.eval_limit = 1000
        Args.seed = 200
        #     Args.seed = [200, 300, 400, 500, 600, 700]

    for deps in sweep:
        # thunk = instr(train, deps, __prefix="pretrain-local-metric", __postfix=f"{Args.local_metric}")
        thunk = instr(eval_planning_policy, deps, __prefix='planning-policy',
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


def planning_policy():
    from uvpn.maze.evaluations import EvalArgs, PlanArgs
    with Sweep(EvalArgs, PlanArgs) as sweep:
        # Args.env_id = "CMazeDiscreteImgIdLess-v0"

        EvalArgs.sample_env_id = "CMazeDiscreteImgIdLess-v0"
        EvalArgs.sample_num_rollouts = 200

        PlanArgs.dump_plans = True

        PlanArgs.neighbor_r = 1.4
        PlanArgs.neighbor_r_min = 0.7
        PlanArgs.prune_r = 0.2

        EvalArgs.latent_dim = 10
        EvalArgs.search_alg = "dijkstra"
        EvalArgs.local_metric = "LocalMetricConvLargeL2"
        EvalArgs.load_local_metric = "/geyang/plan2vec/2020/02-08/neo_plan2vec/" \
                                     "uvpn_image/quick_eval_new_local_metric/" \
                                     "local_metric/hige_loss/lr-sweep/12.24/" \
                                     "LocalMetricConvLargeL2-6e-05/" \
                                     "CMazeDiscreteImgIdLess-v0/12.783076/models/0100/f_lm.pkl"

        EvalArgs.load_adv = '/amyzhang/plan2vec/2020/02-05/neo_plan2vec/uvpn_image/' \
                            'maze_uvpn_image/train-inverse-model-small/11.06/lr-6e-' \
                            '05-seed-400/03.638480/models/0095/adv.pkl'

        EvalArgs.seed = 400
        EvalArgs.num_evals = 2
        EvalArgs.num_eval_rollouts = 1
        EvalArgs.eval_limit = 200
        with sweep.zip:
            EvalArgs.eval_limit = [50, 1000]
            EvalArgs.env_id = ["CMazeDiscreteImgIdLess-v0", "FourRoomDiscreteImgIdLess-v0"]
            # EvalArgs.seed = [200, 300, 400, 500, 600, 700]

    for deps in sweep[1:]:
        # thunk = instr(train, deps, __prefix="pretrain-local-metric", __postfix=f"{Args.local_metric}")
        thunk = instr(eval_planning_policy, deps,
                      poses=[[-0.15, 0.15]],
                      goals=[[-0.15, -0.15]],
                      __prefix="adaptation",
                      __postfix=f"{EvalArgs.local_metric}-r-{PlanArgs.neighbor_r}-rmin-{PlanArgs.neighbor_r_min}/{EvalArgs.env_id}")
        jaynes.run(thunk, )
        config_charts("""
            charts:
            - yKey: success/mean
              xKey: eval_step
            keys:
            - run.status
            - PlanArgs.prune_r
            - PlanArgs.neighbor_r
            - PlanArgs.neighbor_r_min
            """)


if __name__ == '__main__':
    jaynes.config()
    planning_policy()
    jaynes.listen()
