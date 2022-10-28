from termcolor import cprint

if __name__ == "__main__":
    import jaynes
    import pandas as pd
    from plan2vec_experiments import instr, RUN, dir_prefix, config_charts
    from ml_logger import logger

    jaynes.config('vector')

    logger.configure(log_directory=RUN.server, prefix=dir_prefix() + "/analysis")

    parameter_keys = 'Args.n_rollouts',

    with logger.PrefixContext(dir_prefix()):
        metrics = logger.glob("**/metrics.pkl")

    print(*metrics, sep="\n")

    logger.print("key", *parameter_keys, 'success_rate', sep=',\t', file='results.csv')

    for metrics_path in metrics:
        with logger.PrefixContext(dir_prefix()):
            exp_path = '/'.join(metrics_path.split('/')[:-1])

            if "sample_complexity-sweep" in exp_path or "plan2vec" in exp_path:
                key = "plan2vec"

            parameter_values = logger.get_parameters(*parameter_keys, path=exp_path + "/parameters.pkl")
            df = pd.DataFrame(logger.load_pkl(metrics_path))

        try:
            last_success = df['success_rate/mean'][-100:]
            avg_success = last_success.mean()

            logger.print(key, parameter_values, avg_success, sep=',\t', file='results.csv')
        except:
            cprint(exp_path + "fails to load", "red")
