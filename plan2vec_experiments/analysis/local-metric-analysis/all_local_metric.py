import jaynes
from params_proto.neo_hyper import Sweep
from plan2vec_experiments import instr, config_charts

seeds = [i * 100 for i in range(2)]


def img_maze():
    from plan2vec.plan2vec.local_metric_img import main, Args

    with Sweep(Args) as sweep:
        Args.num_epochs = 40
        # Note: using L2 is critical, because other wise the local metric fails
        #  to pick up the [0 - 1] continuity.
        Args.local_metric = "LocalMetricConvLargeL2"
        with sweep.product:
            Args.env_id = [
                # "GoalMassDiscreteImgIdLess-v0",
                # "回MazeDiscreteImgIdLess-v0",
                "CMazeDiscreteImgIdLess-v0",
            ]
            Args.seed = seeds

    for _Args in sweep:
        jaynes.run(instr(main, __postfix=f"img_maze/{Args.local_metric}/{Args.env_id}", **_Args))
        config_charts(path="charts/img_maze.charts.yml")


def rope():
    from plan2vec.plan2vec.local_metric_rope import main, Args

    with Sweep(Args) as sweep:
        Args.num_epochs = 5
        Args.K = 1
        Args.k_fold = None
        with sweep.product:
            Args.seed = seeds

    for _Args in sweep:
        _ = instr(main, __postfix=f"{Args.env_id}/K-({Args.K})", **_Args)
        config_charts(path="charts/rope.charts.yml")
        jaynes.run(_)


def streetlearn():
    from plan2vec.plan2vec.local_metric_streetlearn import main, Args

    seeds = [100]

    with Sweep(Args) as sweep:
        Args.num_epochs = 50
        Args.checkpoint_interval = 100
        Args.lr = 1e-5
        Args.data_path = f"~/fair/streetlearn/processed-data/manhattan-medium"

        with sweep.product:
            # Args.local_metric = ["LocalMetricConvDeep", "ResNet18L2"]
            Args.local_metric = ["ResNet18L2"]
            Args.seed = seeds

    for _Args in sweep:
        _ = instr(main, __postfix=f"{Args.env_id}/{Args.local_metric}/lr-({Args.lr})", **_Args)
        config_charts(path="charts/streetlearn.charts.yml")
        jaynes.run(_)


if __name__ == '__main__':
    jaynes.config()

    img_maze()
    # rope()
    streetlearn()
    jaynes.listen(300)
