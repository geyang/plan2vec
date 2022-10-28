from plan2vec_experiments.analysis_icml_2020 import mlc, stylize, cmap
from uvpn.maze.domains.maze import maze_simpple_testset

start_goals = maze_simpple_testset()


@mlc
def c_maze_goal_render():
    import gym
    import matplotlib.pyplot as plt
    from ml_logger import logger
    import numpy as np
    from ge_world import IS_PATCHED

    assert IS_PATCHED

    env = gym.make("CMazeDiscreteIdLess-v0")
    frames = []
    for start, goal in start_goals:
        env.set_state(np.array([*start, *goal]), qvel=np.zeros(4))
        img = env.render('rgb', width=240, height=240)
        frames.append(img)

    frames = np.array(frames)

    decay = np.exp(- 0.15 * np.arange(frames.shape[0]))
    overlay = 255 - np.max((255 - frames) * decay[:, None, None, None], axis=0).astype(int)

    plt.figure(figsize=(2.4, 2.4), dpi=200)
    plt.title('Goal Dataset', pad=10)
    plt.imshow(overlay)
    plt.axis('off')
    plt.tight_layout()

    logger.savefig("figures/cmaze_test_data.png")
    plt.show()
    plt.close()


@mlc
def c_maze_no_goal():
    import gym
    import matplotlib.pyplot as plt
    from ml_logger import logger
    import numpy as np
    from ge_world import IS_PATCHED

    assert IS_PATCHED

    env = gym.make("CMazeDiscreteIdLess-v0")
    env.set_state(np.array([-0.15, 0.15, 10, 10]), qvel=np.zeros(4))
    img = env.render('rgb', width=240, height=240)

    plt.figure(figsize=(2.4, 2.4), dpi=200)
    plt.title('Goals', pad=10)
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()

    logger.savefig("figures/cmaze_goal_data.png")
    plt.show()
    plt.close()


if __name__ == '__main__':
    stylize()
    # c_maze_goal_render()
    c_maze_no_goal()
