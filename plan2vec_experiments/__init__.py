import inspect
import os
from functools import reduce, partial
from os.path import basename, dirname, abspath, join

import yaml
from termcolor import cprint

with open(os.path.join(os.path.dirname(__file__), ".yours"), 'r') as stream:
    rc = yaml.load(stream, Loader=yaml.BaseLoader)


class RUN:
    from ml_logger import logger
    # noinspection All
    while True:  # sometimes the host is not ready.
        try:
            is_cluster = "cluster" in logger.hostname
            break
        except TypeError:
            import time
            time.sleep(1)
            print('sleep for 1 second waiting for host to be ready')

    counter = 0
    server = "http://54.71.92.65:8081"
    # server = "/checkpoint" if is_cluster else 'http://localhost:8090'
    prefix = f"{rc['username']}/{rc['project']}/{logger.now('%Y/%m-%d')}"


def dir_prefix(depth=-1):
    from ml_logger import logger

    caller_script = abspath(inspect.getmodule(inspect.stack()[1][0]).__file__)
    # note: for scripts in the `plan2vec` module this also works -- b/c we truncate fixed depth.
    script_path = logger.truncate(caller_script, depth=len(__file__.split('/')) - 1)
    prefix = os.path.join(RUN.prefix, script_path)
    return reduce(lambda p, i: dirname(p), range(-depth), prefix)


def config_charts(config_yaml="", path=None):
    from textwrap import dedent
    from ml_logger import logger

    if not config_yaml:

        try:
            caller_script = abspath(inspect.getmodule(inspect.stack()[1][0]).__file__)
            cwd = os.path.dirname(caller_script)
        except:
            cwd = os.getcwd()

        if path is None:
            path = logger.stem(caller_script) + ".charts.yml"
        try:  # first try the namesake chart file
            with open(os.path.join(cwd, path), 'r') as s:
                config_yaml = s.read()
                cprint(f"Found ml-dash config file \n{path}", 'green')
        except:  # do not upload when can not find
            path = ".charts.yml"
            with open(os.path.join(cwd, path), 'r') as s:
                config_yaml = s.read()
            cprint(f"Found ml-dash config file \n{path}", 'green')

    logger.log_text(dedent(config_yaml), ".charts.yml")


def instr(fn, *ARGS, __prefix="", __postfix=None, __no_timestamp=False, __file=False, __silent=False, **KWARGS):
    """
    thunk for configuring the logger. The reason why this is not a decorator is

    :param fn: function to be called
    :param *ARGS: position arguments for the call
    :param __postfix: logging prefix for this run, default to "", where it does not do much.
    :param __no_timestamp: boolean flag to turn off the training timestamp.
    :param __file__: console mode, by-pass file related logging
    :param __silent: do not print
    :param **KWARGS: keyword arguments for the call
    :return: a thunk that can be called without parameters
    """
    import jaynes
    from ml_logger import logger

    if __file:
        caller_script = os.path.join(os.getcwd(), __file)
    else:
        __file = inspect.getmodule(inspect.stack()[1][0]).__file__
        caller_script = abspath(__file)

    # note: for scripts in the `plan2vec` module this also works -- b/c we truncate fixed depth.
    script_path = logger.truncate(caller_script, depth=len(__file__.split('/')) - 1)
    file_stem = logger.stem(script_path)
    file_name = os.path.basename(file_stem)
    PREFIX = join(RUN.prefix, file_stem, __prefix or "", jaynes.RUN.now('%H.%M.%S'), __postfix or "",
                  "" if __no_timestamp else f"{RUN.counter:03d}")
    RUN.counter += 1

    # todo: there should be a better way to log these.
    # todo: we shouldn't need to log to the same directory, and the directory for the run shouldn't be fixed.
    logger.configure(log_directory=RUN.server, prefix=PREFIX, asynchronous=False,  # use sync logger
                     max_workers=4, register_experiment=False)
    logger.upload_file(caller_script)
    # the tension is in between creation vs run. Code snapshot are shared, but runs need to be unique.
    _ = dict()
    if ARGS:
        _['args'] = ARGS
    if KWARGS:
        _['kwargs'] = KWARGS

    logger.log_params(
        run=logger.run_info(status="created", script_path=script_path),
        revision=logger.rev_info(),
        fn=logger.fn_info(fn),
        **_,
        silent=__silent)

    logger.diff(silent=True)

    import jaynes  # now set the job name to prefix
    if jaynes.RUN.config and jaynes.RUN.mode != "local":
        runner_class, runner_args = jaynes.RUN.config['runner']
        if 'name' in runner_args:  # ssh mode does not have 'name'.
            runner_args['name'] = join(__prefix or "", file_name or '', __postfix or '')
        del logger, jaynes, runner_args, runner_class
        if not __file:
            cprint(f'Set up job name', "green")

    def thunk(*args, **kwargs):
        import traceback
        from ml_logger import logger

        assert not (args and ARGS), f"can not use position argument at both thunk creation as well as " \
                                    f"run.\n_args: {args}\nARGS: {ARGS}"

        import os
        logger.configure(log_directory=RUN.server, prefix=PREFIX, register_experiment=False, max_workers=10)
        logger.log_params(host=dict(hostname=logger.hostname),
                          run=dict(status="running", startTime=logger.now(),
                                   # job_id=logger.job_id  # learnfair is stuck at ml_logger 0.4.52.
                                   job_id=os.getenv('SLURM_JOB_ID', None)
                                   ))

        import time
        try:
            _KWARGS = {**KWARGS}
            _KWARGS.update(**kwargs)

            results = fn(*(args or ARGS), **_KWARGS)

            logger.log_line("========= execution is complete ==========")
            logger.log_params(run=dict(status="completed", completeTime=logger.now()))
            logger.flush()
            time.sleep(3)
        except Exception as e:
            tb = traceback.format_exc()
            with logger.SyncContext():  # Make sure uploaded finished before termination.
                logger.log_text(tb, filename="traceback.err")
                logger.log_params(run=dict(status="error", exitTime=logger.now()))
                logger.log_line(tb)
                logger.flush()
            time.sleep(3)
            raise e

        return results

    return thunk
