from plan2vec.plan2vec.maze_uvpn_image import Args, train

if __name__ == '__main__':
    import jaynes
    from params_proto.neo_hyper import Sweep
    from plan2vec_experiments import instr, config_charts
    from ml_logger import logger

    jaynes.config()

    with Sweep(Args) as sweep:
        Args.env_id = "CMazeDiscreteImgIdLess-v0"
        # Args.env_id = "GoalMassDiscreteImgIdLess-v0"
        Args.latent_dim = 2
        Args.num_rollouts = 400
        Args.search_alg = "dijkstra"
        Args.num_epochs = 100
        Args.checkpoint_overwrite = False
        Args.checkpoint_interval = 5
        with sweep.product:
            Args.local_metric = [
                "LocalMetricConv",  # 3'24
                # "LocalMetricConvLarge",  # 3:56
                # "LocalMetricConvXL",  # 5:05
                # "LocalMetricConvDeep",  # 5:46
                # "ResNet18Stacked"  # 17:50
            ]
            Args.seed = [200, 300, 400]

    for deps in sweep:
        thunk = instr(train, deps, __prefix="pretrain-local-metric", __postfix=f"{Args.local_metric}")
        jaynes.run(thunk, )
        config_charts("""
            charts:
            - yKey: loss/mean
              xKey: epoch
            - yKey: d/mean
              xKey: epoch
            - yKey: d_bar/mean
              xKey: epoch
            - yKey: d_null/mean
              xKey: epoch
            - yKey: d_shuffle/mean
              xKey: epoch
            keys:
            - run.status
            - Args.lr
            - Args.n_rollouts
            """)
        logger.log_text(__doc__, "README.md")

    jaynes.listen(600)
