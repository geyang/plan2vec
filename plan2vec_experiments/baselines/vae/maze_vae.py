from plan2vec.vae.image_vae import train

if __name__ == "__main__":
    import jaynes
    from plan2vec_experiments import instr, config_charts

    jaynes.config("vector-gpu")
    for latent_dim in [2]:
        for env_id in [
            'GoalMassDiscreteImgIdLess-v0',
            '回MazeDiscreteImgIdLess-v0',
            'CMazeDiscreteImgIdLess-v0',
        ]:
            for lr in [3e-5, 1e-4, 3e-4, 1e-3]:
                _ = instr(train, env_id=env_id, lr=lr, latent_dim=latent_dim, n_workers=4,
                          __postfix=f"dim-({latent_dim})/{env_id}/lr-({lr})")
                config_charts()
                jaynes.run(_)

    jaynes.listen()
