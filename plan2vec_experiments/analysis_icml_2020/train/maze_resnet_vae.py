from plan2vec.vae.resnet_vae import train

if __name__ == '__main__':
    import jaynes
    from plan2vec_experiments import instr, config_charts

    thunk = instr(train, n_epochs=100)
    config_charts(train.__doc__)
    jaynes.config()
    jaynes.run(thunk)

    jaynes.listen()
