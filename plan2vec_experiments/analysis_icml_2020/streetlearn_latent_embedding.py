from plan2vec_experiments.analysis_icml_2020 import mlc, stylize, cmap


@mlc  # no GPU support
def dump_streat_views():
    from ml_logger import logger

    env_id = "manhattan-large"
    from streetlearn import StreetLearnDataset
    path = f"~/fair/streetlearn/processed-data/{env_id}"

    pad = 0.1
    d = dataset = StreetLearnDataset(path, view_size=(128, 64), view_mode='omni-rgb', )
    d.select_bbox(-73.997, 40.726, 0.01, 0.008)
    d.show_blowout("NYC-large", show=True)

    a = d.bbox[0] + d.bbox[2] * pad, d.bbox[1] + d.bbox[3] * pad
    b = d.bbox[0] + d.bbox[2] * (1 - pad), d.bbox[1] + d.bbox[3] * (1 - pad)
    (start, _), (goal, _) = d.locate_closest(*a), d.locate_closest(*b)

    logger.log_image(d.images[start], key="figures/sl_start.png")
    logger.log_image(d.images[goal], key="figures/sl_goal.png")

    logger.log_images(d.images[[start, goal]], n_cols=1, n_rows=2, key="figures/sl_stacked.png")


@mlc  # no GPU support
def plot_v_fn_embedding(checkpoint, title, filename, latent_dim=2):
    import torch
    from torch_utils import Eval, torchify
    from ml_logger import ML_Logger, logger
    from plan2vec.models.resnet import ResNet18L2

    assert latent_dim == 2, "only support 2-d due to time limit"

    from plan2vec_experiments import RUN
    read_logger = ML_Logger(RUN.server)

    Φ = ResNet18L2(input_dim=1, latent_dim=latent_dim)
    logger.print('loading checkpoint', color="green")
    read_logger.load_module(Φ, checkpoint)

    @torchify(dtype=torch.float32, input_only=True)
    def embed_fn_eval(*_):
        with Eval(Φ), torch.no_grad():
            return Φ.embed(*_)

    from plan2vec.plotting.streetlearn.embedding_image_streetlearn \
        import cache_images, visualize_embedding_2d_image, visualize_embedding_3d_image
    from plan2vec.plan2vec.streetlearn_plan2vec import StreetLearnGraph, load_streetlearn

    env_id = "manhattan-large"
    dataset, start, goal = load_streetlearn(f"~/fair/streetlearn/processed-data/{env_id}")
    graph = StreetLearnGraph(dataset, 2.4e-4, p=2)
    cache_images(graph.all_images, graph.all_positions, graph.lat_correction)

    figsize = (2.1, 2.4)

    # plotting the oracle
    visualize_embedding_2d_image(
        title, embed_fn_eval, axis_off=True, cmap=cmap, alpha=0.9, size=2,
        figsize=figsize, dpi=300,
        filename=filename)


@mlc  # no GPU support
def load_vae_from_checkpoint(checkpoint, title, filename, latent_dim=2, input_dim=1):
    import torch
    from torch_utils import Eval, torchify
    from ml_logger import ML_Logger
    from plan2vec.models.resnet import ResNet18Coord

    from plan2vec_experiments import RUN
    read_logger = ML_Logger(RUN.server)

    Φ = ResNet18Coord(input_dim, num_classes=latent_dim * 2)
    read_logger.load_module(Φ, checkpoint)

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
def load_embedding_data(data_path, title, filename):
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
def load_dqn_from_checkpoint(checkpoint, title, filename, latent_dim=2, input_dim=1):
    import torch
    from torch_utils import Eval, torchify
    from ml_logger import ML_Logger
    from plan2vec.mdp.models import ResNet18CoordL2Q, ResNet18CoordQ

    from plan2vec_experiments import RUN
    read_logger = ML_Logger(RUN.server)

    policy_net = ResNet18CoordL2Q(input_dim, 8, latent_dim, p=2)
    read_logger.load_module(policy_net, checkpoint)

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


if __name__ == '__main__':
    stylize()
    dump_streat_views()
    # plot_v_fn_embedding(title="UVPN-StreetLearn",
    #                     checkpoint="/amyzhang/plan2vec/2020/02-02/analysis_icml_2020/train/"
    #                                "streetlearn_uvpn/sweep-L_p/16.19/lr(3e-06)-rs-(2000)-p2/"
    #                                "manhattan-large/21.149056/models/04000/Φ.pkl",
    #                     filename=f"figures/sl_embed_uvpn_{2}d.png")
    plot_v_fn_embedding(title="2D Embedding",
                        checkpoint="/geyang/plan2vec/2020/04-05/neo_plan2vec/streetlearn/streetlearn_plan2vec/16.53.06/manhattan-large-lr(1e-05)-p2-ResNet18L2/000/models/10000-Φ.pkl",
                        filename=f"figures/sl_embed_{2}d.png")
    plot_v_fn_embedding(title="3D Embedding",
                        checkpoint="/geyang/plan2vec/2020/04-05/neo_plan2vec/streetlearn/streetlearn_plan2vec/16.53.06/manhattan-large-lr(1e-05)-p2-ResNet18L2/000/models/10000-Φ.pkl",
                        filename=f"figures/sl_embed_{3}d.png")
