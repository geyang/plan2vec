"""
you should be able to launch using this script.
"""

nodes = ['gpu012', 'gpu052', "gpu043", "gpu028"]


def try_gym():
    from ml_logger import logger
    from termcolor import cprint
    import torch

    cprint(f"the hostname is {logger.hostname}", "white")

    # try:
    #     import gym
    #     env = gym.make('Reacher-v2')
    #     env.reset()
    #     cprint(f"host {logger.hostname} is fine", "green")
    #
    for name in nodes:
        if logger.hostname.startswith(name):
            cprint(f"host {logger.hostname} has failed", "yellow")
            print('running while loop to block the node')
            i = 0
            from time import time

            t0 = time()
            while True:
                a = torch.ones(10000, 10000).to('cuda')
                b = torch.randn_like(a).to('cuda')
                c = (a * b).mean()
                if time() - t0 > 300000:
                    print("have been five minutes, let's release the node.")
                    break
                if i % 10000:
                    print(c)
                i += 1


if __name__ == '__main__':
    import jaynes

    jaynes.config()
    for i in range(len(nodes) * 4):  # 4 GPUs per node.
        jaynes.run(try_gym)
    jaynes.listen()
