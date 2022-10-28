from params_proto.neo_proto import ParamsProto, Proto
from params_proto.neo_hyper import Sweep


class Args(ParamsProto):
    seed = 0
    env_id = "StreetLearn"
    lr = 0.1


class DEBUG(ParamsProto):
    print_deps = False


def launch(deps=None, **kwargs):
    import sys

    print(*sys.path, sep="\n")

    import torchvision.models as models
    print(models)

    import gym

    env = gym.make('Reacher-v2')
    print(env)

    from ml_logger import logger
    import time
    time.sleep(1)

    Args._update(**kwargs)
    logger.log_params(Args=vars(Args))

    if DEBUG.print_deps:
        print(f"deps is:", deps, sep='\n')


if __name__ == "__main__":
    import jaynes

    jaynes.config(verbose=True)

    DEBUG.print_deps = True

    jaynes.run(launch, **vars(DEBUG))

    jaynes.listen(0)

