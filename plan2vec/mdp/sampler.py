from collections import defaultdict
import numpy as np


def path_gen_fn(env, goal_act_fn, obs_key, goal_key, all_keys=None, start_reset=False):
    """
    Generator function for the path data. This one outputs the log-likelihood, value and baseline.

    Usage:

    .. code:: python

        s = path_gen_fn(...)
        timesteps = 100
        paths = s.send(timesteps)

        assert "acs" in paths
        assert "obs" in paths

    :param env: A parallel env, with first index being the batch
    :param goal_act_fn: has the signature `act`, returns batched actions for each observation (in the batch)
    :param obs_key: the key for the observation (for the policy)
    :param goal_key: the key for the goal (also for the policy)
    :param all_keys: all keys that we want to return in the path object
    :param start_reset: boolean flag for resting on each generator start.
    :return: dimension is Size(timesteps, n_envs, feature_size)
    """
    # todo: use a default dict for these data collection. Much cleaner.
    assert goal_key is not None, "todo: need to implement for goal_key is None."
    all_keys = all_keys or (obs_key, goal_key)

    timesteps = yield
    obs, dones = env.reset(), [False] * env.num_envs

    paths = defaultdict(list)
    while True:
        paths['obs'] = defaultdict(list)
        paths['next'] = defaultdict(list)
        # do NOT use this if environment is parallel env.
        if start_reset:  # note: mostly useless.
            obs, dones = env.reset(), [False] * env.num_envs
        for _ in range(timesteps):
            for k in all_keys:
                paths['obs'][k].append([item[k] for item in obs])
            actions = goal_act_fn(paths['obs'][obs_key][-1], paths['obs'][goal_key][-1])
            paths['acs'].append(actions)
            obs, rewards, dones, info = env.step(actions)

            paths['rewards'].append(rewards)
            paths['dones'].append(dones)
            # note: next now returns full observation stack.
            for k in all_keys:
                paths['next'][k].append([info_i['next'][k] if done_i else ob_i[k]
                                         for info_i, ob_i, done_i in zip(info, obs, dones)])

            # In multiworld, `info` contains the entire observation. Processing these
            # will take way too much time. So we don't do that.
            _suc = [_['success'] for _ in info if 'success' in _]
            if _suc:
                paths['info.successes'].append(_suc)
            _dist = [_['dist'] for _ in info if 'dist' in _]
            if _dist:
                paths['info.dists'].append(_dist)

        # The TimeLimit env wrapper "dones" the env when time limit
        # has been reached. This is technically not correct.
        # if has vf and not done. Discounted infinite horizon.
        timesteps = yield {
            k: {_k: np.array(_v) for _k, _v in v.items()} if isinstance(v, dict) else np.array(v)
            for k, v in paths.items()
        }
        paths.clear()
