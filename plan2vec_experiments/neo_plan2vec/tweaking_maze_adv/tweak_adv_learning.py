"""
Maze Environment, Tweaking the learning rate for the advantage function.

"""
from plan2vec.plan2vec.maze_plan2vec import Args, train

if __name__ == '__main__':
    import jaynes
    from params_proto.neo_hyper import Sweep
    from plan2vec_experiments import instr, config_charts
    from ml_logger import logger

    jaynes.config()

    # start_epoch = 801
    # load_global_metric = f"/geyang/plan2vec/2020/01-23/plan2vec/maze_plan2vec/advantage-centroid-loss/18.03/dijkstra-lr-1e-05-ctr-0.001/56.741100/models/{start_epoch - 1:04d}/Φ.pkl"

    with Sweep(Args) as sweep:
        Args.start_epoch = 801
        Args.load_global_metric = f"/geyang/plan2vec/2020/01-23/plan2vec/maze_plan2vec/advantage-centroid-loss/18.03/dijkstra-lr-1e-05-ctr-0.001/56.741100/models/{800:04d}/Φ.pkl"

        Args.env_id = "GoalMassDiscreteImgIdLess-v0"
        Args.latent_dim = 2
        Args.limit = 4
        Args.num_rollouts = 400
        # Args.search_alg = ["dijkstra", "a_star", "heuristic"]
        Args.search_alg = "dijkstra"
        Args.seed = 20
        Args.adv_bp_scale = 1
        with sweep.product:
            Args.adv_lr = [1e-5]  # , 6e-6, 3e-6, 1e-6, 3e-7, 1e-7]

    for deps in sweep:
        # thunk = instr(train, deps, __prefix="load_pretrained_Φ-larger-net-rmsprop",
        #               __postfix=f"adv_lr-{Args.adv_lr}-bp_scale-{Args.adv_bp_scale}")
        thunk = instr(train, deps, __prefix="load_pretrained_Φ-larger-quick-inspection",
                      __postfix=f"zero-baseline-soft-adv_lr-{Args.adv_lr}-bp_scale-{Args.adv_bp_scale}")
        jaynes.run(thunk, )
        config_charts("""
charts:
- yKey: act_adv/mean
  xKey: epoch
  yDomain: [-0.01, 0.05]
- yKey: adv_target/mean
  xKey: epoch
  yDomain: [-0.01, 0.05]
- yKey: adv_values/mean
  xKey: epoch
  yDomain: [-0.01, 0.05]
- yKey: adv_bp/mean
  xKey: epoch
  yDomain: [-0.01, 0.05]
  
- yKey: act_loss/mean
  xKey: epoch
  yDomain: [0, 0.0035]
- yKey: back_pressure_loss/mean
  xKey: epoch
  yDomain: [0, 0.0035]
  
- yKey: success/mean
  xKey: epoch
# - yKey: cost/mean
#   xKey: epoch
# - type: image
#   glob: "**/sample_graph.png"
# - type: image
#   glob: "figures/embedding/*.png"
keys:
- run.status
- Args.n_rollouts
            """)
        logger.log_text(__doc__, "README.md")

    jaynes.listen(600)
