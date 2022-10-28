def experiment():
    from env_data.generate_samples import Args, main
    Args.env_id = "InvertedPendulum-v0"
    Args.policy = "null"
    Args.width = 28
    Args.height = 28
    if jaynes.RUN.mode in [None, "local"]:
        Args.n_rollouts = 100
    else:
        Args.num_envs = 40

    Args.name = Args.env_id + "/validate"
    Args.n_rollouts = 200
    main()

    Args.name = Args.env_id
    Args.n_rollouts = 2000
    # move out run under both
    main()
    import time
    time.sleep(5)  # wait for data to be sent to logging server

    from plan2vec.vae.maze_vae import Args, train
    Args.env_name = "InvertedPendulum-v0"
    Args.act_dim = 1
    Args.obs_dim = 3
    Args.latent_dim = 3
    Args.n_epochs = 2000
    # Args.batch_size = 100
    train()

    # # load the data
    # from ml_logger import logger
    # saved_weight_path = "/debug/cpc-belief/2018-12-15/point_mass_vae/22.17-457450/weights/0400_encoder.pkl"
    # encoder, = logger.load_pkl(saved_weight_path)


if __name__ == "__main__":
    import jaynes
    from plan2vec_experiments import instr

    jaynes.config("vector-gpu")
    jaynes.run(instr(experiment, ))
    jaynes.listen()
