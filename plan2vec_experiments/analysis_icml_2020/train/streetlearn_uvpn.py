from params_proto.neo_hyper import Sweep
from plan2vec.plan2vec.streetlearn_plan2vec import Args, DEBUG, r_scale_dict, train


def small():
    # ENVS = ["manhattan-tiny", "manhattan-small", "manhattan-medium", "manhattan-large"]
    # ENVS = ["manhattan-tiny", "manhattan-small", "manhattan-large"]
    # ENVS = ["manhattan-small", "manhattan-large"]

    with Sweep(Args, DEBUG) as sweep:
        Args.lr = 1e-5
        Args.batch_size = 100
        Args.global_metric = "ResNet18L2"
        Args.optim_batch_size = 16
        # Args.search_alg = "dijkstra"
        Args.visualization_interval = 10
        with sweep.product:
            # with sweep.zip:
            #     Args.num_epochs = [1500, 5000]
            #     Args.checkpoint_interval = [500, 1000]
            #     Args.r_scale = [70, 500]
            #     Args.env_id = ["manhattan-small", "manhattan-large"]
            with sweep.zip:
                Args.num_epochs = [2000, 4000]
                Args.checkpoint_interval = [2000, 1000]
                Args.r_scale = [70, 200, ]
                Args.env_id = ["manhattan-small", "manhattan-medium"]

            # Args.metric_p = [1, 2]
            Args.search_alg = ["dijkstra", "a_star", ]
            Args.metric_p = [2]
            Args.seed = [400, 500]

    launch(sweep, "small")


def large():
    # ENVS = ["manhattan-tiny", "manhattan-small", "manhattan-medium", "manhattan-large"]
    # ENVS = ["manhattan-tiny", "manhattan-small", "manhattan-large"]
    # ENVS = ["manhattan-small", "manhattan-large"]

    with Sweep(Args, DEBUG) as sweep:
        Args.lr = 3e-6
        Args.batch_size = 100
        Args.global_metric = "ResNet18L2"
        Args.optim_batch_size = 16
        Args.search_alg = "dijkstra"
        Args.visualization_interval = 10
        with sweep.product:
            # with sweep.zip:
            #     Args.num_epochs = [1500, 5000]
            #     Args.checkpoint_interval = [500, 1000]
            #     Args.r_scale = [70, 500]
            #     Args.env_id = ["manhattan-small", "manhattan-large"]
            with sweep.zip:
                Args.num_epochs = [14000]
                Args.checkpoint_interval = [2000]
                # Args.r_scale = [500]
                Args.env_id = ["manhattan-large"]

            # Args.metric_p = [1, 2]
            # Args.search_alg = ["dijkstra", "a_star", ]
            Args.r_scale = [4000, 8000]
            Args.metric_p = [2]
            Args.search_alg = ["dijkstra"]
            Args.seed = [600, 700]

    launch(sweep, "sweep-L_p")


def launch(sweep, prefix):
    import jaynes

    from plan2vec_experiments import instr, config_charts
    from ml_logger import logger

    jaynes.config()

    for deps in sweep:
        thunk = instr(train, deps,
                      __prefix=prefix,
                      __postfix=f"{Args.env_id}-lr({Args.lr})-rs-({Args.r_scale})-p{Args.metric_p}")
        jaynes.run(thunk, )
        config_charts("""
        charts:
        - yKey: loss/mean
          xKey: epoch
        - yKey: value/mean
          xKey: epoch
        - yKey: cost/mean
          xKey: epoch
        - type: image
          glob: "**/sample_graph.png"
        - type: image
          glob: "figures/embedding/*.png"
        - type: image
          glob: "debug/*.png"
        keys:
        - run.status
        - Args.n_rollouts
        """)
        logger.log_text(__doc__, "README.md")


if __name__ == '__main__':
    import jaynes

    # small()
    large()

    jaynes.listen(600)
