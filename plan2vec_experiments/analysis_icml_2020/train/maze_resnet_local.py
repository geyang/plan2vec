import jaynes
from params_proto.neo_hyper import Sweep
from plan2vec_experiments import instr, config_charts

seeds = [i * 100 for i in range(2)]


def img_maze():
    from plan2vec.plan2vec.local_metric_img import main, Args

    with Sweep(Args) as sweep:
        Args.num_epochs = 1
        Args.latent_dim = 2
        Args.local_metric = "ResNet18CoordL2"
        with sweep.product:
            Args.env_id = [
                # "GoalMassDiscreteImgIdLess-v0",
                # "回MazeDiscreteImgIdLess-v0",
                "CMazeDiscreteImgIdLess-v0",
            ]
            Args.seed = seeds

    for deps in sweep:
        thunk = instr(main, deps,  __prefix="debug-saving")
        jaynes.run(thunk)
        config_charts("""
        charts:
          - {xKey: epoch, yKey: accuracy-0.5/mean, yDomain: [0.8, 1]}
          - {xKey: epoch, yKey: accuracy-1.1/mean, yDomain: [0.8, 1]}
          - {xKey: epoch, yKey: accuracy-1.5/mean, yDomain: [0.8, 1]}
        keys:
          - run.status
          - Args.lr
          - Args.seed
          - {metrics: accuracy/mean}
        """)


if __name__ == '__main__':
    jaynes.config()

    img_maze()
    jaynes.listen(300)

