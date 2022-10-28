from plan2vec.plan2vec.streetlearn_plan2vec import Args, DEBUG, r_scale_dict, train

if __name__ == '__main__':
    import jaynes
    from params_proto.neo_hyper import Sweep
    from plan2vec_experiments import instr, config_charts
    from ml_logger import logger

    jaynes.config()

    # ENVS = ["manhattan-tiny", "manhattan-small", "manhattan-medium", "manhattan-large"]
    ENVS = ["manhattan-small"]

    with Sweep(Args, DEBUG) as sweep:

        DEBUG.supervise_value = True

        Args.num_epochs = 10000
        Args.input_dim = 3
        Args.view_mode = "omni-rgb"
        Args.global_metric = "ResNet18L2"
        Args.optim_batch_size = 16
        Args.search_alg = "dijkstra"
        Args.visualization_interval = 10
        with sweep.product:
            Args.lr = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
            Args.metric_p = [1, 2]
            with sweep.zip:
                # Args.r_scale = [r_scale_dict[env] for env in ENVS]
                Args.r_scale = [500]
                Args.env_id = ENVS

    for deps in sweep:
        thunk = instr(train, deps, __prefix="PN-triplet-fix-target-sign",
                      __postfix=f"{Args.env_id}-lr({Args.lr})-supervised-p{Args.metric_p}-{Args.global_metric}")
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
