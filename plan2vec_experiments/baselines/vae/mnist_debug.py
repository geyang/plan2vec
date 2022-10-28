from plan2vec.vae.cnn_vae import train
from plan2vec_experiments import instr

if __name__ == "__main__":
    import jaynes

    jaynes.config("vector-gpu")
    for lr in [3e-5, 1e-4, ]:
        jaynes.run(instr(train, lr=lr, n_workers=4))

    jaynes.listen()
