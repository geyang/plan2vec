from plan2vec_experiments import instr
from ml_logger import logger


def train_and_gmo():
    from plan2vec.plan2vec.plan2vec_state import Args, main

    # local_metric_exp_path = "episodeyang/plan2vec-experiments/2019/03-30/local_metric_kernel_loss/13.00/02.895857"
    # local_metric_exp_path = "episodeyang/plan2vec-experiments/2019/03-30/local_metric_kernel_loss/12.46/06.004803"
    local_metric_exp_path = "episodeyang/plan2vec-experiments/2019/03-30/local_metric_kernel_loss/13.58/10.911041"

    Args.num_epochs = 5000
    Args.optim_steps = 1
    Args.lr = 1e-4
    Args.visualization_interval = 1
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"
    main(**vars(Args))


if __name__ == "__main__":
    _ = instr(train_and_gmo, logger.now(f"{logger.stem(__file__)}/{logger.now('%H.%M/%S.%f')}"))
    _()
