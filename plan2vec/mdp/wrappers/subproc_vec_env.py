import numpy as np
from multiprocessing import Process, Pipe, get_context
from baselines.common.vec_env import CloudpickleWrapper


# make synchronous interface for get call

def worker(_, remote, env):
    _.close()
    env = env.x() if hasattr(env, 'x') else env()
    while True:
        # noinspection PyBroadException
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    info['next'] = ob
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == 'get':
                remote.send(getattr(env, data))
            elif cmd == 'close':
                remote.close()
                break  # this terminates the process.
            else:
                data = data or dict()
                args = data.get('args', tuple())
                kwargs = data.get('kwargs', dict())
                if hasattr(env, cmd):
                    _ = getattr(env, cmd)(*args, **kwargs)
                else:
                    _ = getattr(env.unwrapped, cmd)(*args, **kwargs)
                remote.send(_)

        except Exception:
            remote.close()
            break


class SubprocVecEnv:
    reset_on_done = True

    def __init__(self, env_fns, verbose=False):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        ctx = self.multiprocess_context = get_context("fork")

        self.remotes, work_remotes = zip(*[Pipe() for _ in range(self.num_envs)])
        self.ps = [ctx.Process(target=worker, args=(_, remote, CloudpickleWrapper(env_fn)))
                   for _, remote, env_fn in zip(self.remotes, work_remotes, env_fns)]

        _ = zip(self.ps, work_remotes)
        if verbose:
            from tqdm import tqdm
            _ = tqdm(_, desc=f'fork SubprocVecEnv chldProc')

        for p, r in _:
            p.daemon = True  # daemon children are killed when main proc terminate.
            p.start()
            r.close()

        self.first = self.remotes[0]
        self.first.send(('get', 'action_space'))
        self.action_space = self.first.recv()
        self.first.send(('get', 'observation_space'))
        self.observation_space = self.first.recv()
        self.first.send(('get', 'spec'))
        self.spec = self.first.recv()

    def fork(self, n):
        from copy import copy
        _self = copy(self)
        _self.remotes = _self.remotes[:n]
        return _self

    def call_sync(self, fn_name, *args, **kwargs):
        """
        synchronize the return value from all workers.

        :param fn_name:
        :param args:
        :param kwargs:
        :return:
        """
        _ = fn_name, dict(args=args, kwargs=kwargs)
        for remote in self.remotes:
            remote.send(_)
        return np.stack([remote.recv() for remote in self.remotes])

    def get(self, key):
        raise NotImplementedError('need to decide for self.first or all.')

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        obs, rews, dones, infos = zip(*[remote.recv() for remote in self.remotes])
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def render(self, *args, **kwargs):
        return self.call_sync('render', *args, **kwargs)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        return self.call_sync('reset')

    def sample_task(self, *args, **kwargs):
        return self.call_sync('sample_task', *args, **kwargs)

    def reset_task(self):
        self.call_sync('reset_task')

    def close(self):
        """looks bad: mix sync and async handling."""
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def __del__(self):
        self.close()

    def first_call_sync(self, fn_name, *args, **kwargs):
        self.first.send((fn_name, dict(args=args, kwargs=kwargs)))
        return self.first.recv()


if __name__ == "__main__":
    def make_env():
        import gym
        return gym.make('Reacher-v2')


    parallel_envs = SubprocVecEnv([make_env for i in range(6)])
    obs = parallel_envs.reset()
    assert len(obs) == 6, "the original should have 6 envs"

    render_envs = parallel_envs.fork(4)
    # note: here we test the `fork` method, useful for selectiong a sub-batch for rendering purposes.
    obs = render_envs.reset()
    assert len(obs) == 4, "the forked env should have only 4 envs."

    print('test complete.')
