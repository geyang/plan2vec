import numpy as np

from plan2vec.mdp.wrappers.subproc_vec_env import SubprocVecEnv


# now deprecated to use embedding_2d_img.py instead. Move this to the vis function.
def imaginator(env: SubprocVecEnv, fn, low=-0.25, high=0.25, n=21, goal_n=7, width=28, height=28, action=None,
               device=None):
    import torch
    xs = ys = np.linspace(low, high, n)
    xs, ys = np.meshgrid(xs, ys)

    xy_img = {}
    for x, y in zip(xs.flatten(), ys.flatten()):
        env.first_call_sync('set_state', np.array([x, y, 0.3, 0.3]), np.array([0] * 4))
        img = env.first_call_sync('render', 'rgb', width=width, height=height).transpose(2, 0, 1).mean(
            axis=0, keepdims=True) / 255
        xy_img[(x, y)] = torch.Tensor(img).unsqueeze(0).to(device)

    g_ = np.linspace(low, high, goal_n)
    g_xs, g_ys = np.meshgrid(g_, g_)
    for x, y in zip(g_xs.flatten(), g_ys.flatten()):
        env.first_call_sync('set_state', np.array([x, y, 0.3, 0.3]), np.array([0] * 4))
        img = env.first_call_sync('render', 'rgb', width=width, height=height).transpose(2, 0, 1).mean(
            axis=0, keepdims=True) / 255
        xy_img[(x, y)] = torch.Tensor(img).unsqueeze(0).to(device)

    def _(*args, **kwargs):
        r = fn(
            *[torch.cat([xy_img[tuple(i)] for i in arg]) for arg in args],
            **{k: torch.Tensor(v).to(device) for k, v in kwargs.items()})
        if action is not None:
            return r[:, action].cpu().detach().numpy()
        else:
            return r.cpu().detach().numpy()

    return _


def imaginator_maze(env_id, fn, n=21, width=64, height=64, device=None):
    from collections import defaultdict
    from ge_world import gym
    import torch

    env = gym.make(env_id)
    image_grid, xys = [], []
    xy_img = defaultdict(lambda: None)
    for x in np.linspace(env.obj_low, env.obj_high, n):
        for y in np.linspace(env.obj_low, env.obj_high, n):
            _ = (x, y)
            xys.append(_)
            if env.is_good_goal(_):
                env.reset_model(x=_)
                image = env.render('rgb', width=width, height=height)
                image_grid.append(image)
                xy_img[_] = image

    env.close()

    def _(xys):
        # todo: need to line up the images
        # mask = [image_grid[tuple(xy)] for xy in xys]
        r = fn(torch.tensor(image_grid, device=device, dtype=torch.float32))
        return r.cpu().detach().numpy()

    return _
