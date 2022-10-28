# only works with SubprocVecEnv.
import numpy as np

from collections import defaultdict


def render_gen_fn(env, act_fn, obs_key, goal_key, width=200, height=150, reset_on_done=False):
    """
    sample and image generator function for both batched and single environments.

    returns the camera rendering under the `view` key.

    The batched envs require resets when completed. Use `reset_on_done`
    flag to make that happen.

    In the end, it is not worth it to keep the nested dict for `info`. So we flatten it.

    yields "ac", "vpred", "ob", "rew", "new", "view", *"info.deep_keys"

    :param pi:
    :param env:
    :param stochastic:
    :param width:
    :param height:
    :param reset_on_done: should only be True when using single, non-batched environments. Otherwise
        would lead to a ndarray trueful evaluation error.
    :return:
    """
    ob = env.reset()
    while True:
        ob_dict = dict_concat(ob)
        ac = act_fn(ob_dict[obs_key], ob_dict[goal_key])
        _ob, rew, new, _ = env.step(ac)
        view = env.render("rgb", width=width, height=height)
        yield dict(ac=ac, ob=ob_dict, rew=rew, new=new, view=view,
                   **{"info." + k: v for k, v in dict_concat(_, axis=0).items()})
        # lazy evalation is a great thing -- Ge
        if reset_on_done and new:
            ob = env.reset()
        else:
            ob = _ob


# only works with SubprocVecEnv.
def sample_gen_fn(env, act_fn, obs_key, goal_key, reset_on_done=False):
    """
    sample generator function for both batched and single environments.

    The batched envs require resets when completed. Use `reset_on_done`
    flag to make that happen.

    yields "ac", "vpred", "ob", "rew", "new", *"info.deep_keys"

    In the end it is not worth the increased complexity to handle nested dicts for
    the infos. A flat dictionary with key spaces is now the standard.

    :param pi:
    :param env:
    :param stochastic:
    :param reset_on_done: should only be True when using single, non-batched environments. Otherwise
        would lead to a ndarray trueful evaluation error.
    :return: yields "ac", "vpred", "ob", "rew", "new", "info"
    """
    # always reset at the beginning.
    ob = env.reset()
    while True:
        ob_dict = dict_concat(ob)

        ac = act_fn(ob_dict[obs_key], ob_dict[goal_key])
        _ob, rew, new, _ = env.step(ac)
        # flatten info keys.
        yield dict(ac=ac, ob=ob_dict, rew=rew, new=new,
                   **{"info." + k: v for k, v in dict_concat(_, axis=0).items()})
        # lazy evaluation is a great thing -- Ge
        if reset_on_done and new:
            ob = env.reset()
        else:
            ob = _ob


def dict_concat(paths, axis=1):
    store = defaultdict(list)
    for p in paths:
        for k, v in p.items():
            store[k].append(v)
    _ = {}
    for k, v in store.items():
        # try:
        #     _[k] = np.concatenate(v, axis=axis)
        # except ValueError as e:
        _[k] = np.array(v)
    return _


# removed task_id
def make_env(env_id, env_seed, *wrapper_fns):
    """
    Simple environment factory.

    :param env_id: id of the environment you want to make
    :param env_seed: the seed for the environment
    :return:
    """
    from ge_world import gym

    def _f():
        env = gym.make(env_id)
        for fn in wrapper_fns:
            env = fn(env)
        env.seed(seed=env_seed)
        return env

    return _f


# todo: measure the performance of this
def samples_to_path(samples):
    """
    Agnostic to the shape of the observation and action samples.

    Automatically handles "success" => "successes", appending "es".

    :param samples: Array(dict['ob', 'ac', 'rew', ...], ...)
    :return: dict['obs', 'acs', 'rews', 'info.successes', ...]
    """
    import pandas as pd, numpy as np
    df = pd.DataFrame(samples)
    paths = {f"{k}es" if k.endswith('s') else f"{k}s": np.stack(v, 0)
             for k, v in df.to_dict(orient="series").items()}
    # note: remove special handling of nested dictionaries
    # shape = list(paths['infos'].shape)
    # df = pd.DataFrame.from_records(paths['infos'].flatten())
    # paths['infos'] = {k: np.array(v).reshape([*shape, *v.shape[1:]])
    #                   for k, v in df.to_dict(orient="series").items()}
    return paths


def unbatch_policy(policy):
    _policy = lambda: None
    _policy.act = lambda ob, stoch: [
        r if r is None else r.squeeze(0)
        for r in policy.act(ob.reshape(1, -1), stoch)
    ]
    return _policy
