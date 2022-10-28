from plan2vec.supervised_distance.rms_pair_distance import train

if __name__ == "__main__":
    import jaynes
    from ml_logger import logger
    from plan2vec_experiments import instr, config_charts

    jaynes.config("vector-gpu")
    ts = logger.now("%H.%M.%S")
    for n in range(1, 12):
        d = n * 0.04
        _ = instr(train, pair_distance=d, __postfix=f"{ts}/d-{d:0.2f}", __no_timestamp=True)
        config_charts()
        jaynes.run(_)
        # jaynes.listen()
    jaynes.listen()
