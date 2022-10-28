"""
Sweeping over a few different maze environments, to
collect baseline measurements on how well the advantage learning works.

"""
from plan2vec.plan2vec.maze_plan2vec import Args, train

if __name__ == '__main__':
    import jaynes
    from params_proto.neo_hyper import Sweep
    from plan2vec_experiments import instr, config_charts
    from ml_logger import logger

    jaynes.config()

    with Sweep(Args) as sweep:
        Args.limit = 2
        Args.num_rollouts = 800
        Args.search_alg = "dijkstra"
        Args.env_id = "CMazeDiscreteImgIdLess-v0"

        Args.start_epoch = 801
        Args.num_epochs = 1501

        Args.eval_interval = 100
        Args.checkpoint_interval = 100
        visualization_interval = None

        Args.adv_lr = 1e-5

        with sweep.product:
            with sweep.zip:
                Args.latent_dim = [3, 2, 2]
                Args.load_global_metric = [
                    f"/geyang/plan2vec/2020/01-24/neo_plan2vec/tweaking_maze_adv/maze_env_sweep/18.28/14.942307/models/{800:04d}/Φ.pkl"
                    # f"/geyang/plan2vec/2020/01-24/neo_plan2vec/tweaking_maze_adv/maze_env_sweep/18.28/05.420062/models/{540:04d}/Φ.pkl"
                    # f"/geyang/plan2vec/2020/01-24/neo_plan2vec/tweaking_maze_adv/maze_env_sweep/18.24/12.339428/models/{800:04d}/Φ.pkl",
                    # f"/geyang/plan2vec/2020/01-24/neo_plan2vec/tweaking_maze_adv/maze_env_sweep/18.24/11.300457/models/{800:04d}/Φ.pkl"
                ]
            Args.adv_lr = [1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
            # Args.env_id = ["GoalMassDiscreteImgIdLess-v0", "CMazeDiscreteImgIdLess-v0"]

    for deps in sweep:
        thunk = instr(train, deps,
                      __prefix="reintroduce-backpressure",
                      __postfix=f"adv_lr-{Args.adv_lr}-num_rollouts-{Args.num_rollouts}"
                      )
        jaynes.run(thunk, )
        config_charts("""
charts:
- yKey: adv_loss/mean
  xKey: epoch
  yDomain: [0, 0.0035]
- yKey: adv_act/mean
  xKey: epoch
  yDomain: [-0.05, 0.001]
- yKey: adv_target/mean
  xKey: epoch
  yDomain: [-0.05, 0.001]
- yKey: adv_values/mean
  xKey: epoch
  yDomain: [-0.05, 0.001]

- yKey: success/mean
  xKey: epoch
# - yKey: cost/mean
#   xKey: epoch
# - type: image
#   glob: "**/sample_graph.png"
- type: image
  glob: "figures/embedding/*.png"
keys:
- run.status
- Args.n_rollouts
            """)
        logger.log_text(__doc__, "README.md")

    jaynes.listen(600)
