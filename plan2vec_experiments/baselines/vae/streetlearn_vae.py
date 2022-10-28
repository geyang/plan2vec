from plan2vec.vae.streetlearn_vae import train

# todo: set up the sampling.

if __name__ == "__main__":
    import jaynes
    from plan2vec_experiments import instr, config_charts

    # jaynes.config("vector-gpu")
    # for lr in [3e-5, 1e-4, 3e-4, 3e-3]:
    #     _ = thunk(train, lr=lr, n_workers=4, )
    #     config_charts()
    #     jaynes.run(_)
    #
    # jaynes.listen()

    jaynes.config("vector-gpu")
    for dim in [2, 3]:
        for env_id in ['manhattan-tiny', 'manhattan-small', 'manhattan-medium', 'manhattan-large']:
            data_path = f"~/fair/streetlearn/processed-data/{env_id}"
            for lr in [3e-5, 1e-4, 3e-4, 3e-3]:
                _ = instr(train, env_id=env_id, lr=lr, n_workers=4,
                          latent_dim=dim, data_path=data_path,
                          __postfix=f"dim-({dim})/{env_id}/lr-({lr})")
                config_charts()
                jaynes.run(_)

    jaynes.listen()
