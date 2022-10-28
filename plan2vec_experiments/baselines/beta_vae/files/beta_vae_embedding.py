from plan2vec.vae.image_vae import Args, train

if __name__ == "__main__":
    from params_proto.neo_hyper import Sweep

    from plan2vec_experiments import instr, config_charts

    with Sweep(Args) as sweep:
        Args.lr = 1e-4
        Args.latent_dim = 3
        with sweep.product:
            Args.beta = [1e-2, 1e-1, 1]
            Args.env_id = [
                'GoalMassDiscreteImgIdLess-v0',
                '回MazeDiscreteImgIdLess-v0',
                'CMazeDiscreteImgIdLess-v0',
            ]

    import jaynes

    jaynes.config()
    for override in sweep:
        _ = instr(train, n_workers=4, **override,
                  __postfix=f"dim-({Args.latent_dim})/{Args.env_id}/beta-({Args.beta})")
        config_charts("""
            keys:
              - Args.lr
              - Args.latent_dim
              - Args.seed
            charts:
              - yKey: loss/mean
                xKey: epoch
              - type: image
                glob: "**/vae_*.png"
              - type: image
                glob: "**/embed_*.png"
            """)
        jaynes.run(_)
    jaynes.listen()
