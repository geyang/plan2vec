from plan2vec.vae.cnn_vae import train
from plan2vec_experiments import instr

if __name__ == "__main__":
    import jaynes

    jaynes.config("vector-gpu")
    for lr in [1e-5, 3e-5, 1e-4, 3e-4, 6e-4, 1e-3, 3e-3, 6e-3, 1e-2]:
        jaynes.run(instr(train, lr=lr, n_workers=4))

    jaynes.listen()
