import numpy.ma as ma


def visualize_neighbors(xs, ns, key=None):
    """
    Visualizing points, and their neighbors in the dataset

    :param xs: Size(batch_n, 2)
    :param ns: Size(batch_n, k, 2), k being the number of neighbors to show
    :param key: The path to save the figure to
    :return:
    """
    import matplotlib.pyplot as plt

    assert len(xs) == len(ns), "state samples and be neighbors need to have the same length"

    DPI = 300
    title = "neighbors"

    plt.figure(figsize=(3, 3), dpi=DPI, )
    plt.title(title)

    k = xs.shape[0]
    for i, (x, neighbors) in enumerate(zip(xs, ns)):
        alpha = (i + 1) / (k + 1)
        # note: marker size = points/inch * actual axial size.
        plt.plot(x[0], x[1], "o", c="black", alpha=alpha, markersize=DPI * 0.015 * 2, mec="none")
        for n in neighbors:
            plt.plot([x[0], n[0]], [x[1], n[1]], "-", c='black', alpha=alpha, linewidth=3, mec='none')

    plt.xlim(-0.3, 0.3)
    plt.ylim(-0.3, 0.3)
    plt.gca().set_aspect('equal')

    if key is None:
        plt.show()
    else:
        from ml_logger import logger
        logger.savefig(key)
    plt.close()


def visualize_start_goal(starts, goals, key=None):
    """
    Visualizing the start and the goals

    :param starts: Size(batch_n, 2)
    :param goals: Size(batch_n, 2)
    :param key: The path to save the figure to
    :return:
    """
    import matplotlib.pyplot as plt

    assert len(starts) == len(goals), "starts and goals need to have the same length"

    DPI = 300
    title = "Start and Goals"

    plt.figure(figsize=(3, 3), dpi=DPI, )
    plt.title(title)

    k = goals.shape[0]
    for i, (x, g) in enumerate(zip(starts, goals)):
        alpha = (i + 1) / (k + 1)
        # note: marker size = points/inch * actual axial size.
        plt.plot(g[0], g[1], "o", c="black", alpha=alpha, markersize=DPI * 0.015 * 2, mec="none")
        plt.plot([x[0], g[0]], [x[1], g[1]], "-", c='black', alpha=alpha, linewidth=3, mec='none')

    plt.xlim(-0.3, 0.3)
    plt.ylim(-0.3, 0.3)
    plt.gca().set_aspect('equal')

    if key is None:
        plt.show()
    else:
        from ml_logger import logger
        logger.savefig(key)
    plt.close()


def visualize_latent_plans(xs, goals, done, key=None):
    """
    Visualizing the sample trajectories in a 2-dimensional domain

    We use the done flags to

    :param xs: Size(B, n, 2)
    :param goals: Size(B, n, 2)
    :param done: Size(B, n, 2)
    :param key: The path to save the figure to
    :return:
    """
    import matplotlib.pyplot as plt

    colors = ['#49b8ff', '#66c56c', '#f4b247']

    DPI = 300
    title = "Trajectory Distribution"

    plt.figure(figsize=(3, 3), dpi=DPI, )
    plt.title(title)

    n, k, *_ = xs.shape
    for i in range(k):
        alpha = (i + 1) / (k + 1)
        # alpha = 0.7
        x, g, d = ma.array(xs[:, i]), ma.array(goals[:, i]), done[:, i]
        x[d] = ma.masked
        g[d] = ma.masked
        c = colors[i % len(colors)]
        plt.plot(g[:, 0], g[:, 1], "o", c="red", alpha=0.7 / (n + 1), markersize=DPI * 0.04 * 2, mec="none")
        plt.plot(ma.array(x[:, 0]), ma.array(x[:, 1]), 'o-', c="#23aaff", alpha=alpha, linewidth=2,
                 markersize=DPI * 0.01 * 2, mec="none")

    plt.xlim(-0.3, 0.3)
    plt.ylim(-0.3, 0.3)
    plt.gca().set_aspect('equal')

    if key is None:
        plt.show()
    else:
        from ml_logger import logger
        logger.savefig(key)
    plt.close()


def visualize_trajectories_2d(paths, key=None):
    """
    Visualizing the sample trajectories in a 2-dimensional domain

    :param paths: the dictionary of the sampled dataset
    :param key: The path to save the figure to
    :return:
    """
    import matplotlib.pyplot as plt

    DPI = 300
    title = "Trajectory Distribution"

    plt.figure(figsize=(3, 3), dpi=DPI, )
    plt.title(title)

    goals = paths['obs']["goal"][:1]
    k = goals.shape[1]
    # note: marker size = points/inch * actual axial size.
    plt.plot(goals[:, :, 0], goals[:, :, 1], "o", c="red", alpha=0.7, markersize=DPI * 0.04 * 2, mec="none")

    trajs = paths['obs']["x"]
    # todo: ues different color for different trajectories
    # note: marker size = points/inch * actual axial size.
    plt.plot(trajs[:, :, 0], trajs[:, :, 1], 'o-', c="#23aaff", alpha=10 / k, linewidth=2,
             markersize=DPI * 0.01 * 2, mec="none")

    plt.xlim(-0.3, 0.3)
    plt.ylim(-0.3, 0.3)
    plt.gca().set_aspect('equal')

    if key is None:
        plt.show()
    else:
        from ml_logger import logger
        logger.savefig(key)
    plt.close()


def visualize_skewed_trajectories(paths, key=None):
    """
    Visualizing the sample trajectories in a 2-dimensional domain

    :param paths: the dictionary of the sampled dataset
    :param key: The path to save the figure to
    :return:
    """
    import matplotlib.pyplot as plt

    DPI = 300
    title = "Trajectory Distribution"

    plt.figure(figsize=(3, 3), dpi=DPI, )
    plt.title(title)

    goals = paths['obs']["goal"][:1]
    k = goals.shape[1]
    # note: marker size = points/inch * actual axial size.
    if k < 10:
        plt.plot(goals[:, :, 0], goals[:, :, 1], "o", c="gray", alpha=0.1, markersize=DPI * 0.04 * 2, mec="none")

    trajs = paths['obs']["x"]
    k = trajs.shape[1]

    colors = ['#23aaff'] if k > 10 else ['#49b8ff', '#66c56c', '#f4b247']

    from ge_world.c_maze import good_goal
    good = np.array([good_goal(_) for _ in trajs.reshape(-1, 2)]).reshape(trajs.shape[:2])
    # todo: ues different color for different trajectories
    # note: marker size = points/inch * actual axial size.
    for i, traj in enumerate(np.swapaxes(trajs, 0, 1)):
        plt.plot(traj[:, 0], traj[:, 1], '-', c=colors[i % len(colors)], alpha=0.5 if k < 10 else (10. / k),
                 linewidth=2)

    for i, (traj, g) in enumerate(zip(np.swapaxes(trajs, 0, 1), np.swapaxes(good, 0, 1))):
        c = list(np.where(g, colors[i % len(colors)], "red"))
        plt.scatter(traj[:, 0], traj[:, 1], s=DPI * 0.05 * 2, c=c, marker='o', alpha=0.8 if k < 10 else (10. / k),
                    linewidth=2, edgecolors="none")

    plt.xlim(-0.3, 0.3)
    plt.ylim(-0.3, 0.3)
    plt.gca().set_aspect('equal')

    if key is None:
        plt.show()
    else:
        from ml_logger import logger
        logger.savefig(key)
    plt.close()


if __name__ == "__main__":
    from tqdm import trange
    import numpy as np
    from plan2vec.mdp.sampler import path_gen_fn

    from plan2vec.mdp.helpers import make_env
    from plan2vec.mdp.wrappers.subproc_vec_env import SubprocVecEnv

    from ge_world import IS_PATCHED

    assert IS_PATCHED is True, "need patch"

    envs = SubprocVecEnv([make_env('CMazeDiscreteIdLess-v0', i) for i in trange(500)])

    random_pi = lambda ob, goal, *_: np.random.randint(0, 8, size=[len(ob)])

    rev_a_dict = {(-0.5, -0.5): 0,
                  (-0.5, 0): 1,
                  (-0.5, 0.5): 2,
                  (0, -0.5): 3,
                  (0, 0.5): 4,
                  (0.5, -0.5): 5,
                  (0.5, 0): 6,
                  (0.5, 0.5): 7}


    def homing_pi(ob, goal):
        act = map(tuple, 0.5 * abs(np.array(goal) - np.array(ob)) / (np.array(goal) - np.array(ob)))
        return [rev_a_dict[a] for a in act]


    eps = 0.1


    def greedy_pi(ob, goal):
        if np.random.rand(1) < eps:
            return random_pi(ob, goal)
        return homing_pi(ob, goal)


    # servo_pi = lambda ob, goal, *_: rev_a_dict[
    #     tuple(0.5 * abs(np.array(goal) - np.array(ob)) / (np.array(goal) - np.array(ob)))]

    rand_path_gen = path_gen_fn(envs, greedy_pi, "x", "goal", all_keys=['x', 'goal'])
    next(rand_path_gen)
    p = rand_path_gen.send(50)
    # a, b = p['obs']['x'][:2, 0]
    # b - a
    # print(p['x'])

    # visualize_trajectories_2d(p, f"./figures/CMaze trajectories (fixed).png")
    # visualize_trajectories_2d(p, f"./figures/CMaze trajectories (homing).png")
    # visualize_trajectories_2d(p, f"./figures/CMaze trajectories (greedy {eps}).png")
    visualize_skewed_trajectories(p, f"figures/CMaze rejected goals (500).png")
    # visualize_skewed_trajectories(p)
