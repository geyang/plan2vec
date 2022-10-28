from plan2vec.plan2vec.maze_plan2vec import Args, train


def a_star_run():
    from params_proto.neo_hyper import Sweep
    from plan2vec_experiments import instr, config_charts
    from ml_logger import logger

    with Sweep(Args) as sweep:
        # Args.env_id = "GoalMassDiscreteImgIdLess-v0"
        Args.env_id = "CMazeDiscreteImgIdLess-v0"
        Args.latent_dim = 3

        Args.adv_after = None

        Args.checkpoint_interval = 400

        with sweep.product:
            Args.search_alg = ["a_star", "heuristic"]
            Args.seed = range(10, 15)

    import jaynes
    for deps in sweep:
        thunk = instr(train, deps,
                      __prefix="cost-vs-learning", __postfix=Args.search_alg)
        jaynes.run(thunk, )
        config_charts("""
            charts:
            - yKey: adv_target/mean
              xKey: epoch
            - yKey: adv_values/mean
              xKey: epoch
            - yKey: success/mean
              xKey: epoch
            - yKey: cost/mean
              xKey: epoch
            - type: image
              glob: "**/sample_graph.png"
            - type: image
              glob: "figures/embedding/*.png"
            keys:
            - run.status
            - Args.n_rollouts
            """)
        logger.log_text(__doc__, "README.md")


def default_run():
    from params_proto.neo_hyper import Sweep
    from plan2vec_experiments import instr, config_charts
    from ml_logger import logger

    with Sweep(Args) as sweep:
        # Args.env_id = "GoalMassDiscreteImgIdLess-v0"
        Args.env_id = "CMazeDiscreteImgIdLess-v0"
        Args.latent_dim = 2

        Args.adv_lr = 1e-4
        Args.adv_after = 1200
        Args.num_epochs = 3000

        Args.checkpoint_interval = 400

        with sweep.product:
            Args.search_alg = ["dijkstra", "a_star", "heuristic"]
            Args.seed = [100 * i for i in range(10, 15)]

    import jaynes
    for deps in sweep:
        thunk = instr(train, deps,
                      __prefix="larger_adv_lr",
                      __postfix=f"{Args.search_alg}/adv_lr-{Args.adv_lr}")
        jaynes.run(thunk, )
        config_charts("""
            charts:
            - yKey: adv_target/mean
              xKey: epoch
            - yKey: adv_values/mean
              xKey: epoch
            - yKey: success/mean
              xKey: epoch
            - yKey: cost/mean
              xKey: epoch
            - type: image
              glob: "**/sample_graph.png"
            - type: image
              glob: "figures/embedding/*.png"
            keys:
            - run.status
            - Args.n_rollouts
            """)
        logger.log_text(__doc__, "README.md")


if __name__ == '__main__':
    import jaynes

    jaynes.config()
    default_run()
    a_star_run()

    jaynes.listen(600)
