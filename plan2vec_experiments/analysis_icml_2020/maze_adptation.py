from plan2vec_experiments.analysis_icml_2020 import mlc, stylize, cmap


@mlc
def four_room_render():
    import gym
    from ml_logger import logger
    import numpy as np
    from ge_world import IS_PATCHED

    assert IS_PATCHED

    env = gym.make("CMazeDiscreteIdLess-v0")
    env.set_state(np.array([-0.18, 0.15, 10, 10]), np.zeros(4))
    img = env.render('rgb', width=240, height=240)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(2.4, 2.4), dpi=200)
    plt.title('Original Maze', pad=10)
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()

    # logger.savefig("figures/original_maze_render.png")
    plt.show()

    env = gym.make("FourRoomDiscreteIdLess-v0")
    env.set_state(np.array([-0.1, 0.12, 10, 10]), np.zeros(4))
    img = env.render('rgb', width=240, height=240)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(2.4, 2.4), dpi=200)
    plt.title('Insert Walls', pad=10)
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()

    # Note: to generate this graph, change four-room-arena.xml to non-transparent for the walls.
    # logger.savefig("figures/changed_maze_render.png")
    plt.show()

    env = gym.make("FourRoomDiscreteIdLess-v0")
    env.set_state(np.array([10, 10, 10, 10]), np.zeros(4))
    img = env.render('rgb', width=240, height=240)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(2.4, 2.4), dpi=200)
    plt.title('Model Update', pad=10)
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()

    # Note: to generate this graph, change four-room-arena.xml to non-transparent for the walls.
    logger.savefig("figures/four_room_render.png")
    plt.show()


@mlc
def render_plans(exp_path):
    from plan2vec_experiments import RUN
    from ml_logger import logger, ML_Logger
    import matplotlib.pyplot as plt

    loader = ML_Logger(RUN.server, prefix=exp_path)

    steps = [19, 56, 95, 108]
    colors = ["black", "#FF931E", "#7AC943", "#23aaff", ]

    from ml_logger import logger
    import matplotlib.pyplot as plt
    plt.figure(figsize=(2.4, 2.4), dpi=200)
    plt.title('Pruned Edges', pad=10)
    for step, color in [*zip(steps, colors)][::-1]:

        data, = loader.load_pkl(f"plan_dumps/{step:04}_data.pkl")

        a, b = data['removed_edges']

        for a_, b_ in zip(a, b):
            plt.plot([a_[0], b_[0]], [a_[1], b_[1]], color=color)
        # plt.axis('off')
        plt.ylim(-0.26, 0.26)
        plt.xlim(-0.26, 0.26)
    plt.tight_layout()
    logger.savefig(key=f"figures/pruned_edges.pdf")
    loader.savefig(key=f"removed_edges.pdf")
    loader.savefig(key=f"removed_edges.png")
    plt.show()



if __name__ == '__main__':
    four_room_render()
    render_plans("geyang/plan2vec/2020/02-07/neo_plan2vec/uvpn_image/"
                 "maze_uvpn_image/adaptation/00.35/LocalMetricConvLarge-r-1.0-rmin-0.5/"
                 "FourRoomDiscreteImgIdLess-v0/08.661368")
