import numpy as np


def k_index(env):
    """
    add a k timestep index to the observation. Making value prediction and other
    modeling significantly easier. Should work with both single and batched environments, but not tested
    on the latter.

    :param env:
    :return:
    """
    import gym
    gym.logger.set_level(40)
    _step = env.step
    _reset = env.reset

    assert isinstance(env.observation_space, gym.spaces.box.Box), 'we only support the box observation atm.'
    ks = None  # the step counter

    def obfilt(obs):
        nonlocal ks
        if ks is None:
            ks = np.zeros(*obs.shape[:-1], 1)
        return np.concatenate([obs, ks], axis=-1)

    def step(vac):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        nonlocal ks
        obs, rewards, news, infos = _step(vac)
        _obs = obfilt(obs)
        ks = (1 - news) + ks * (1 - news)
        return _obs, rewards, news, infos

    def reset():
        nonlocal ks
        obs = _reset()
        ks = None  # clear the step counter
        return obfilt(obs)

    env.step = step
    env.reset = reset

    obs_space = env.observation_space
    env.observation_space = gym.spaces.box.Box(
        np.concatenate([obs_space.low, [0]]),
        np.concatenate([obs_space.high, [env.spec.max_episode_steps or None]]),  # note: magic numbers are evil --Ge
    )

    return env


if __name__ == "__main__":
    import gym

    env = gym.make('Reacher-v2')
    print(env.observation_space)
    assert env.observation_space.shape == (11,), 'reacher is 11 dimensional'
    k_env = k_index(env)
    assert env.observation_space.shape == (12,), 'k_index adds 1 to the observation space'
    obs = k_env.reset()
    assert obs.shape == (12,), 'observation should agree with the type'
