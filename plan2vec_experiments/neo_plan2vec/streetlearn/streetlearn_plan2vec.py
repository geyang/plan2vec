from plan2vec.plan2vec.streetlearn_plan2vec import Args, DEBUG, r_scale_dict, train

# 1. how does different metric affect things?
# 2. 2D versus 3D.

def generalization():
    import jaynes
    from params_proto.neo_hyper import Sweep
    from plan2vec_experiments import instr, config_charts
    from ml_logger import logger

    jaynes.config()

    with Sweep(Args, DEBUG) as sweep:
        Args.batch_size = 100
        Args.global_metric = "ResNet18L2"
        Args.optim_batch_size = 16
        Args.search_alg = "dijkstra"
        Args.visualization_interval = 10

        Args.latent_dim = 3

        # # switch to small
        # Args.env_id = "manhattan-small"
        # Args.r_scale = 700
        # Args.metric_p = 1.2
        # Args.lr = 3e-4
        # Args.num_epochs = 2000

        # # medium
        # Args.env_id = "manhattan-medium"
        # Args.r_scale = 2000
        # Args.metric_p = 1.2
        # Args.lr = 1e-5
        # Args.num_epochs = 5000

        # large
        Args.env_id = "manhattan-large"
        Args.r_scale = 4000
        Args.metric_p = 1.2
        Args.lr = 1e-5
        Args.num_epochs = 10_000

        # DEBUG.generalization_sampling = False

        with sweep.product:
            # Args.lr = [1e-5, 3e-5, 1e-4]
            Args.seed = [200, 300, 400]
            DEBUG.generalization_sampling = [True, False]

    for deps in sweep:
        thunk = instr(train, deps,
                      __prefix="generalization",
                      __postfix=f"{Args.env_id}-lr({Args.lr})-p{Args.metric_p}-{Args.global_metric}/sample-{DEBUG.generalization_sampling}")
        jaynes.run(thunk, )
        config_charts("""
        charts:
        - yKey: loss/mean
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
        logger.log_text(f"""
        # Generalization Experiment

        Separate train/test tasks, inspect embedding, evaluate planning performance.

        """, "README.md")

    jaynes.listen(600)


def dimension_comparison():
    import jaynes
    from params_proto.neo_hyper import Sweep
    from plan2vec_experiments import instr, config_charts
    from ml_logger import logger

    jaynes.config()

    with Sweep(Args, DEBUG) as sweep:
        Args.batch_size = 100
        Args.global_metric = "ResNet18L2"
        Args.optim_batch_size = 16
        Args.search_alg = "dijkstra"
        Args.visualization_interval = 10
        Args.num_epochs = 5000
        Args.env_id = "manhattan-medium"
        Args.r_scale = 2000
        with sweep.product:
            Args.latent_dim = [2, 3]
            Args.metric_p = [1.2, 1.5, 2]

    for deps in sweep[-3:]:
        thunk = instr(train, deps,
                      __prefix="dimension_compare",
                      __postfix=f"{Args.env_id}-lr({Args.lr})-p{Args.metric_p}-{Args.global_metric}")
        jaynes.run(thunk, )
        config_charts("""
        charts:
        - yKey: loss/mean
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
        logger.log_text(f"""
        # Compare b/w Different Dimension

        This one uses `p={Args.metric_p}`.
        
        """, "README.md")

    jaynes.listen(600)


def high_dim_baseline():
    import jaynes
    from params_proto.neo_hyper import Sweep
    from plan2vec_experiments import instr, config_charts
    from ml_logger import logger

    jaynes.config()

    with Sweep(Args, DEBUG) as sweep:
        Args.batch_size = 100
        Args.global_metric = "ResNet18L2"
        Args.optim_batch_size = 16
        Args.search_alg = "dijkstra"
        Args.visualization_interval = 10
        Args.num_epochs = 5000
        Args.env_id = "manhattan-medium"
        Args.r_scale = 2000
        with sweep.zip:
            Args.latent_dim = [10, 20]

    for deps in sweep:
        thunk = instr(train, deps,
                      __prefix="high-d-baseline",
                      __postfix=f"{Args.env_id}-lr({Args.lr})-d{Args.latent_dim}-{Args.global_metric}")
        jaynes.run(thunk, )
        config_charts("""
        charts:
        - yKey: loss/mean
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
        logger.log_text(f"""
        # Use 10-dimensional latent space, for better performance during planning

        This one uses `latent_dim={Args.latent_dim}`.
        
        """, "README.md")

    jaynes.listen(600)


def p_comparison():
    import jaynes
    from params_proto.neo_hyper import Sweep
    from plan2vec_experiments import instr, config_charts
    from ml_logger import logger

    jaynes.config()

    with Sweep(Args, DEBUG) as sweep:
        Args.batch_size = 100
        Args.global_metric = "ResNet18L2"
        Args.optim_batch_size = 16
        Args.search_alg = "dijkstra"
        Args.visualization_interval = 10
        Args.num_epochs = 5000
        Args.env_id = "manhattan-medium"
        Args.r_scale = 2000
        with sweep.zip:
            Args.metric_p = [1, 1.05, 1.1, 1.2, 1.5, 2]

    for deps in sweep:
        thunk = instr(train, deps,
                      __prefix="p-compare",
                      __postfix=f"{Args.env_id}-lr({Args.lr})-p{Args.metric_p}-{Args.global_metric}")
        jaynes.run(thunk, )
        config_charts("""
        charts:
        - yKey: loss/mean
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
        logger.log_text(f"""
        # Compare b/w Different Metric p.

        This one uses `p={Args.metric_p}`.
        
        """, "README.md")

    jaynes.listen(600)


def all_streetlear():
    import jaynes
    from params_proto.neo_hyper import Sweep
    from plan2vec_experiments import instr, config_charts
    from ml_logger import logger

    jaynes.config(verbose=True)

    with Sweep(Args, DEBUG) as sweep:
        Args.global_metric = "ResNet18L2"
        Args.optim_batch_size = 16
        Args.search_alg = "dijkstra"
        Args.visualization_interval = 10
        Args.metric_p = 1.2
        # with sweep.zip:
        #     Args.metric_p = [1.2, 1.2, 1.5, 2]
        #     Args.batch_size = [20, 100, 100, 100]
        #     Args.lr = [1e-4, 3e-4, 1e-5, 1e-5]
        #     Args.num_epochs = [500, 2000, 5000, 10_000]  # 5_000 would be sufficient
        #     Args.r_scale = [200, 700, 2000, 4000]
        #     Args.env_id = ["manhattan-tiny",
        #                    "manhattan-small",
        #                    "manhattan-medium",
        #                    "manhattan-large", ]
        with sweep.zip:
            Args.metric_p = [1.1, 1.2, 1.3]
            Args.batch_size = [100]
            Args.lr = [1e-5]
            Args.num_epochs = [10_000]  # 5_000 would be sufficient
            Args.r_scale = [4000]
            Args.env_id = ["manhattan-large", ]
            Args.seed = [200, 300, 400]

    for deps in sweep[-1:]:
        thunk = instr(train, deps,
                      __postfix=f"{Args.env_id}-lr({Args.lr})-p{Args.metric_p}-{Args.global_metric}")
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

    jaynes.listen(600)


if __name__ == '__main__':
    # all_streetlear()
    # p_comparison()
    # dimension_comparison()
    # high_dim_baseline()
    generalization()
