import numpy as np


def vec_normalize(envs, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99):
    ob_rms = RunningMeanStd(shape=envs.observation_space.shape) if ob else None
    ret_rms = RunningMeanStd(shape=()) if ret else None
    ret = np.zeros(envs.num_envs)
    gamma = gamma

    _step = envs.step
    _reset = envs.reset

    def step(vac):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        nonlocal ret, ret_rms, cliprew
        obs, rewards, news, infos = _step(vac)
        _info = dict(reward_mean=rewards.mean(), reward_std=rewards.std())
        ret = ret * gamma + rewards
        obs = _obfilt(obs)
        if ret_rms:
            ret_rms.update(ret)
            rewards = np.clip(rewards / np.sqrt(ret_rms.var), -cliprew, cliprew)
        return obs, rewards, news, _info

    def _obfilt(obs):
        nonlocal clipob
        if ob_rms:
            ob_rms.update(obs)
            obs = np.clip((obs - ob_rms.mean) / np.sqrt(ob_rms.var), -clipob, clipob)
            return obs
        else:
            return obs

    def reset():
        """
        Reset all environments
        """
        obs = _reset()
        return _obfilt(obs)

    envs.step = step
    envs.reset = reset

    return envs


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.zeros(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


def test_runningmeanstd():
    for (x1, x2, x3) in [
        (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
        (np.random.randn(3, 2), np.random.randn(4, 2), np.random.randn(5, 2)),
    ]:
        rms = RunningMeanStd(epsilon=0.0, shape=x1.shape[1:])

        x = np.concatenate([x1, x2, x3], axis=0)
        ms1 = [x.mean(axis=0), x.var(axis=0)]
        rms.update(x1)
        rms.update(x2)
        rms.update(x3)
        ms2 = [rms.mean, rms.var]

        assert np.allclose(ms1, ms2)
