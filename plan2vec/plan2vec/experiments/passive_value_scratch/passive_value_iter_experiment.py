from plan2vec_experiments import instr
from ml_logger import logger


def train_and_gmo():
    from plan2vec.plan2vec.point_mass_passive import Args, main

    local_metric_exp_path = "episodeyang/plan2vec-experiments/2019/03-29/local_metric-debug/12.50/58.471057"

    Args.num_epochs = 5000
    Args.optim_epochs = 10
    Args.lr = 1e-2
    Args.visualization_interval = 1
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"
    main(**vars(Args))


if __name__ == "__main__":
    _ = instr(train_and_gmo, logger.now(f"{logger.stem(__file__)}-debug/{logger.now('%H.%M/%S.%f')}"))
    _()
