from plan2vec_experiments import instr
from plan2vec.cnn_vae.cnn_vae import Args, train

if __name__ == "__main__":
    import jaynes
    from ml_logger import logger

    Args.n_epochs = 1000

    Args.env_id = "GoalMassDiscreteIdLess-v0"
    with logger.SyncContext():
        _ = instr(train, logger.stem(__file__), logger.now("%H.%M.%S"))
        _(_Args=vars(Args))
    exit()

    for Args.latent_dim in [2, 4, 10]:
        _ = instr(train, logger.stem(__file__), logger.now("%H.%M.%S"))
        Args.matplotlib_backend = "Agg"
        jaynes.RUN.J = None
        jaynes.config(runner=dict(n_gpu=1, name=_.prefix, partition='dev'),
                      display=logger.now("%f"))
        jaynes.run(_, _Args=vars(Args))
        import time

        time.sleep(1)
    jaynes.listen()
