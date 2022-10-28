from plan2vec_experiments import instr
from ml_logger import logger


def train_and_gmo():
    from plan2vec.plan2vec.plan2vec_state import DEBUG, Args, main

    # local_metric_exp_path = "episodeyang/plan2vec-experiments/2019/03-30/local_metric_kernel_loss/13.00/02.895857"
    # local_metric_exp_path = "episodeyang/plan2vec-experiments/2019/03-30/local_metric_kernel_loss/12.46/06.004803"
    local_metric_exp_path = "episodeyang/plan2vec-experiments/2019/03-30/local_metric_kernel_loss/13.58/10.911041"

    Args.num_epochs = 500
    Args.lr = 3e-5
    Args.gamma = 0.97
    Args.target_update = 0.90
    Args.visualization_interval = 10
    Args.top_k = 20
    Args.H = 20
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"

    # DEBUG.supervised_value_fn = True
    
    main(**vars(Args), _DEBUG=vars(DEBUG), )


if __name__ == "__main__":
    _ = instr(train_and_gmo, logger.now(f"{logger.stem(__file__)}/{logger.now('%H.%M/%S.%f')}"))
    _()
