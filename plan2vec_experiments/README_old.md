# Best Practices for Experiment Logging

Over time we started to log these things as standard practice because they prevent
many problems from happening. For example, we log the diff of the git repo so that
we know what has been changed for each run. 

Here is an example that we can use for all our scripts:

```python
def train(**_Args):
    """Your training function"""
    Args.update(_Args)
    logger.log_params(Args)
    
    # your training stuff after this.
    # ... ...
    
if __name__ == "__main__":
    from ml_logger import logger
    from gmo_experiments import thunk

    ts = logger.now('%H.%M.%S')
    thunk(train, logger.stem(__file__), ts)()
```

The `thunk` is included in our experiment folder. Please take a look at [./__init__.py](./__init__.py).
It logs a few pieces of information:
*locally*:
- revision: git has, git branch name, and git diff of `HEAD`.
- fn_info: the name of the function, its module, file path, etc.

*remotely (when `fn` is actually called)*:
- `hostname` the machine hostname you are running from
- `*args` to the function is automatically logged
- `**kwargs` to the function is also automatically logged.
- what ever you log inside your training function.
```python

def thunk(fn, *praefixa, postfix=None):
    """
    thunk for configuring the logger. The reason why this is not a decorator is 

    :param fn: function to be called
    :param praefixa: prefixes for the logging directory
    :param postfix: postfix for the logging directory -- set to '%f' when left None.
    :return: None
    """
    import os
    from ml_logger import logger

    experiment_prefix = os.path.join(RUN.prefix, *praefixa, postfix or logger.now("%f"))

    logger.configure(log_directory=RUN.server, prefix=experiment_prefix, register_experiment=False)

    logger.log_params(run=logger.run_info(), revision=logger.rev_info(), fn=logger.fn_info(fn))
    logger.diff(silent=True)
    del logger

    def _(*args, **kwargs):
        import traceback
        from ml_logger import logger
        logger.configure(log_directory=RUN.server, prefix=experiment_prefix, register_experiment=False)
        host = dict(hostname=logger.hostname)
        logger.log_params(host=host)
        try:
            fn(*args, **kwargs)
            logger.log_line("========= execution is complete ==========")
        except Exception as e:
            import time
            time.sleep(1)
            tb = traceback.format_exc()
            logger.log_line(tb)
            logger.flush()
            raise e

    return _
```

## Setting Up Your Username for Logging:

Add a `.yours` file to this folder after git pull, with the following content

```yaml
username: your_username
project: plan2vec
```

Our [../gmo_experiments/__init__.py](/__init__.py) automatically
reads the logging prefix from this file. This way we can log under our own namespace
without conflict.



