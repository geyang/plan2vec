from plan2vec_experiments.analysis_icml_2020 import mlc, stylize, cmap


@mlc
def c_maze_render():
    import gym
    from ml_logger import logger
    import numpy as np
    from ge_world import IS_PATCHED

    assert IS_PATCHED

    env = gym.make("CMazeDiscreteIdLess-v0")
    env.set_state(np.array([0.08, 0.125, 10, 10]), np.zeros(4))
    img = env.render('rgb', width=240, height=240)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(2.4, 2.4), dpi=200)
    plt.title('Maze Environment', pad=10)
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()

    logger.savefig("figures/c_maze_render.png")
    plt.show()


@mlc
def ground_truth_color_code():
    from plan2vec.plotting.maze_world.embedding_image_maze \
        import cache_images, oracle, oracle_3d, visualize_embedding_2d_image, visualize_embedding_3d_image

    title = "Color Code"
    env_id = "CMazeDiscreteImgIdLess-v0"

    cache_images(env_id, n=21)

    # plotting the oracle
    visualize_embedding_2d_image(title, oracle, axis_off=True, cmap=cmap, alpha=0.9,
                                 figsize=(2.4, 2.4), dpi=200, filename=f"figures/embed_color_code.png")
    visualize_embedding_3d_image(title, oracle_3d, axis_off=True, cmap=cmap, alpha=0.9,
                                 figsize=(2.4, 2.4), dpi=200, filename=f"figures/embed_color_code_3d.png")


from waterbear import DefaultBear

MEM = DefaultBear(list, graph=None)


def cache_image_goal_GT_length():
    from tqdm import trange
    from plan2vec.plan2vec.maze_plan2vec import Args, sample_trajs, MazeGraph
    from graph_search import dijkstra
    from ml_logger import logger

    if MEM.graph:
        logger.print('Image for distance prediction has already been generated.')

    else:
        logger.print('First generate images for distance prediction', color="yellow")

        env_id = "CMazeDiscreteImgIdLess-v0"
        trajs = sample_trajs(seed=10, env_id=env_id, num_rollouts=400,
                             obs_keys=['x', 'goal', 'img', 'goal_img'], limit=2)

        graph = MazeGraph(trajs, obs_keys=['x', 'img'], r=Args.neighbor_r, r_min=Args.neighbor_r_min, )
        # graph.show()

        MEM.graph = graph

        for i in trange(100, desc="adding GT path-length data",
                        bar_format="\x1b[32m{l_bar}{bar}{r_bar}\x1b[39m"):
            start, goal = graph.sample(2)
            path, ds = dijkstra(graph, start, goal)
            MEM.steps.append(len(ds))
            MEM.path_length.append(sum(ds))
            MEM.start.append(start)
            MEM.goal.append(goal)
            MEM.path.append(path)

        logger.print("Image cache has finished.", color="green")

    return MEM


def compute_distance(metric_fn, short_name):
    mem = cache_image_goal_GT_length()

    a = mem.graph.all_images[mem.start]
    b = mem.graph.all_images[mem.goal]
    ds = metric_fn(a, b)
    mem[short_name] = ds


@mlc  # no GPU support
def plot_value_fn_embedding(checkpoint, title, filename, latent_dim=2, input_dim=1):
    import torch
    from torch_utils import Eval, torchify
    from ml_logger import ML_Logger
    from plan2vec.models.resnet import ResNet18CoordL2

    from plan2vec_experiments import RUN
    read_logger = ML_Logger(RUN.server)

    Φ = ResNet18CoordL2(input_dim, latent_dim=latent_dim)
    read_logger.load_module(Φ, checkpoint)

    @torchify(dtype=torch.float32)
    def d_eval(*_):
        with Eval(Φ), torch.no_grad():
            return Φ(*_).squeeze()

    compute_distance(d_eval, title)

    @torchify(dtype=torch.float32, input_only=True)
    def embed_fn_eval(*_):
        with Eval(Φ), torch.no_grad():
            return Φ.embed(*_)

    from plan2vec.plotting.maze_world.embedding_image_maze \
        import cache_images, visualize_embedding_2d_image, visualize_embedding_3d_image

    # title = f"UVPN-{latent_dim}D"
    env_id = "CMazeDiscreteImgIdLess-v0"
    cache_images(env_id, n=21)

    # plotting the oracle
    if latent_dim <= 3:
        eval(f"visualize_embedding_{latent_dim}d_image")(
            title, embed_fn_eval, axis_off=True, cmap=cmap, alpha=0.9,
            figsize=(2.4, 2.4), dpi=200,
            filename=filename)


@mlc  # no GPU support
def plot_vae_embedding(checkpoint, title, filename, latent_dim=2, input_dim=1):
    import torch
    from torch_utils import Eval, torchify
    from ml_logger import ML_Logger
    from plan2vec.models.resnet import ResNet18Coord

    from plan2vec_experiments import RUN
    read_logger = ML_Logger(RUN.server)

    Φ = ResNet18Coord(input_dim, num_classes=latent_dim * 2)
    read_logger.load_module(Φ, checkpoint)

    @torchify(dtype=torch.float32)
    def d_eval(a, b):
        with Eval(Φ), torch.no_grad():
            return torch.norm(Φ(a) - Φ(b), dim=-1).squeeze()

    compute_distance(d_eval, title)

    @torchify(dtype=torch.float32, input_only=True)
    def embed_fn_eval(*_):
        with Eval(Φ), torch.no_grad():
            return Φ(*_)[:, :latent_dim]  # <== this is because it is VAE.

    from plan2vec.plotting.maze_world.embedding_image_maze \
        import cache_images, oracle, oracle_3d, visualize_embedding_2d_image, visualize_embedding_3d_image

    env_id = "CMazeDiscreteImgIdLess-v0"
    cache_images(env_id, n=21)

    # plotting the oracle
    if latent_dim <= 3:
        eval(f"visualize_embedding_{latent_dim}d_image")(
            title, embed_fn_eval, axis_off=True, cmap=cmap, alpha=0.9,
            figsize=(2.4, 2.4), dpi=200,
            filename=filename)


@mlc  # no GPU support
def plot_embedding_data(data_path, title, filename):
    from ml_logger import logger, ML_Logger
    import numpy as np
    import matplotlib.pyplot as plt
    from plan2vec_experiments import RUN
    read_logger = ML_Logger(RUN.server)

    data, = read_logger.load_pkl(data_path)
    zs = np.stack([data["z_x"], data['z_y']], axis=-1)
    xys = np.stack([data['x'], data['y']], axis=-1)

    plt.figure(figsize=(2.4, 2.4), dpi=200)
    plt.title(title, pad=10)

    plt.gca().set_aspect('equal', 'datalim')
    plt.scatter(zs[:, 0], zs[:, 1], c=(- xys[:, 0] + xys[:, 1]).flatten(),
                cmap=cmap, alpha=0.9, linewidths=0)
    plt.axis('off')
    logger.savefig(filename)
    plt.close()


@mlc  # no GPU support
def plot_q_network_embedding(checkpoint, title, filename, latent_dim=2, input_dim=1):
    import torch
    from torch_utils import Eval, torchify
    from ml_logger import ML_Logger
    from plan2vec.mdp.models import ResNet18CoordL2Q, ResNet18CoordQ

    from plan2vec_experiments import RUN
    read_logger = ML_Logger(RUN.server)

    policy_net = ResNet18CoordL2Q(input_dim, 8, latent_dim, p=2)
    read_logger.load_module(policy_net, checkpoint)

    @torchify(dtype=torch.float32)
    def d_eval(a, b):
        with Eval(policy_net), torch.no_grad():
            return policy_net(a, b).squeeze()

    compute_distance(d_eval, title)

    @torchify(dtype=torch.float32, input_only=True)
    def embed_fn_eval(*_):
        with Eval(policy_net), torch.no_grad():
            # Note:for Q-learning we just pick the first one
            #   in different Q arch., use embedding directly.
            return policy_net.embed(*_)[..., 0]

    from plan2vec.plotting.maze_world.embedding_image_maze \
        import cache_images, visualize_embedding_2d_image, visualize_embedding_3d_image

    env_id = "CMazeDiscreteImgIdLess-v0"
    cache_images(env_id, n=21)

    # plotting the oracle
    if latent_dim <= 3:
        eval(f"visualize_embedding_{latent_dim}d_image")(
            title, embed_fn_eval, axis_off=True, cmap=cmap, alpha=0.9,
            figsize=(2.4, 2.4), dpi=200,
            filename=filename)


@mlc
def plot_distance_scatter_plot():
    from plan2vec_experiments.analysis_icml_2020 import plot_scatter, COLORS
    import pandas as pd
    import numpy as np

    df = pd.DataFrame({
        "steps": MEM.steps,
        "path_length": MEM.path_length,
        "UVPN": np.array(MEM['UVPN-2D']).squeeze(),
        "SPTM": np.array(MEM['NCE']).squeeze(),
        "VAE": np.array(MEM['VAE']).squeeze(),
    })

    stylize()

    # Note: can choose path_length, less quantization.
    xKey, *yKeys = "steps", 'UVPN', 'VAE', 'SPTM'
    size = 10

    import matplotlib.pyplot as plt

    plt.figure(figsize=(5.2, 2), dpi=300)

    for i, yKey in enumerate(yKeys):
        # plt.title(title, pad=15)
        plt.subplot(1, len(yKeys), i + 1)
        s = df[[xKey, yKey]].dropna()

        label = yKey.replace('_', " ")
        xs, ys = s[xKey], s[yKey]
        plt.title(yKey)
        plt.scatter(xs, ys, color=COLORS[i % 4], alpha=0.7, s=size, edgecolor='none', label=label)

        plt.xlabel("Steps")
        if i == 0:
            plt.ylabel("Score")
        else:
            plt.gca().get_yaxis().set_visible(False)
            plt.gca().spines['left'].set_visible(False)

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    from ml_logger import logger
    logger.savefig("figures/distance_prediction.png")
    plt.show()

    xKey, *yKeys = "path_length", 'UVPN', 'VAE', 'SPTM'
    size = 10

    import matplotlib.pyplot as plt

    plt.figure(figsize=(5.2, 2), dpi=300)

    for i, yKey in enumerate(yKeys):
        # plt.title(title, pad=15)
        plt.subplot(1, len(yKeys), i + 1)
        s = df[[xKey, yKey]].dropna()

        label = yKey.replace('_', " ")
        xs, ys = s[xKey], s[yKey]
        plt.title(yKey)
        plt.scatter(xs, ys, color=COLORS[i % 4], alpha=0.7, s=size, edgecolor='none', label=label)

        plt.xlabel("Distance")
        if i == 0:
            plt.ylabel("Score")
        else:
            plt.gca().get_yaxis().set_visible(False)
            plt.gca().spines['left'].set_visible(False)

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    from ml_logger import logger
    logger.savefig("figures/distance_prediction_path_length.png")
    plt.show()


if __name__ == '__main__':
    plot_distance_scatter_plot()
    exit()

if __name__ == '__main__':
    stylize()
    c_maze_render()
    ground_truth_color_code()
    plot_value_fn_embedding(
        title="UVPN-2D",
        checkpoint="/geyang/plan2vec/2020/01-25/neo_plan2vec/tweaking_maze_adv/"
                   "c-maze_tweak/triangular-target-fix-sign/17.00/adv_lr-1e-06-"
                   "num_rollouts-800/25.730736/models/1000/Φ.pkl",
        filename=f"figures/embed_uvpn_{2}d.png")

    plot_value_fn_embedding(
        title="UVPN-3D",
        checkpoint="/geyang/plan2vec/2020/01-25/neo_plan2vec/tweaking_maze_adv/"
                   "c-maze_tweak/own-trunk/12.46/adv_lr-1e-05-bp_scale-0-nun_rol"
                   "louts-2000/28.997011/models/1000/Φ.pkl",
        latent_dim=3,
        filename=f"figures/embed_uvpn_{3}d.png")

    plot_value_fn_embedding(
        title="NCE",
        checkpoint="/geyang/plan2vec/2020/02-01/analysis_icml_2020/train/maze_"
                   "resnet_local/16.12/32.804446/models/local_metric.pkl",
        latent_dim=2,
        filename=f"figures/embed_nce_{2}d.png")

    plot_vae_embedding(
        title="VAE",
        checkpoint="/geyang/plan2vec/2020/02-01/analysis_icml_2020/train/resnet_vae/"
                   "14.12/57.806152/models/encoder.pkl",
        latent_dim=2,
        filename=f"figures/embed_vae_{2}d.png")

    # Note: goal_mass embedding result:
    #  data_path = "/geyang/plan2vec/2019/12-19/analysis-for-rebuttal/graph-sample-" \
    #              "complexity-hard/maze_plan2vec/23.34/n_rollouts-(400)/57.890102/" \
    #              "embedding_data/embed_2d_0550.pkl"

    plot_embedding_data(
        title="Plan2vec",
        data_path="/geyang/plan2vec/2019/12-19/analysis-for-rebuttal/graph-sample-"
                  "complexity-hard/maze_plan2vec/23.50/CMazeDiscreteImgIdLess-v0/"
                  "ResNet18CoordL2/00.130770/embedding_data/embed_2d_0200.pkl",
        filename="figures/embed_plan2vec_2d.png"
    )

    plot_q_network_embedding(
        title="dqn",
        checkpoint=f"/geyang/plan2vec/2020/02-01/analysis_icml_2020/train/maze_resnet"
                   f"_dqn/16.09/GoalMassDiscreteImgIdLess-v0-ResNet18CoordL2Q/"
                   f"40.034439/models/{600:05d}/policy_net.pkl",
        filename=f"figures/embed_dqn_2d.png"
    )

    exit()
