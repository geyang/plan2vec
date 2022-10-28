from params_proto.neo_proto import ParamsProto
import torch
from torch import nn
from tqdm import tqdm

from env_data.stack_loader import StackLoader


class Args(ParamsProto):
    seed = 100
    env_id = "CMazeDiscreteImgIdLess-v0"
    num_rollouts = 400
    obs_keys = "img", "x"  # All observations we want to be able to get from the environment.
    image_key = "img"
    limit = 2  # the limit of each rollout.

    k_fold = None

    input_dim = 1
    act_dim = 2
    obs_dim = 8

    latent_dim = 2

    batch_size = 100
    test_batch_size = 2

    num_epochs = 400
    lr = 1e-4  # 0.01 for SGD
    optimizer = "Adam"

    use_gpu = True

    debug_interval = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(**kwargs):
    """
    # Training a ResNet18 with Coord conv as an VAE.
    # The decoder is a simple DCNN.
    #
    # :param kwargs:
    # :return:

    charts:
    - yKey: loss/mean
      xKey: epoch
    - type: file
      glob: "**/*.png"
    """
    import torch.nn.functional as F
    import numpy as np
    from torch_utils import View
    from ml_logger import logger

    Args._update(**kwargs)
    logger.log_params(Args=vars(Args))

    from plan2vec.models.resnet import ResNet18Coord

    encoder = ResNet18Coord(input_dim=Args.input_dim, num_classes=Args.latent_dim * 2).to(Args.device)

    decoder = nn.Sequential(
        View(Args.latent_dim, 1, 1),
        nn.ConvTranspose2d(Args.latent_dim, 128, kernel_size=4, stride=1),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, kernel_size=6, stride=2),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, kernel_size=6, stride=2),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
        nn.ReLU(),
        nn.ConvTranspose2d(32, Args.input_dim, kernel_size=5, stride=1),
        nn.Sigmoid(),
    ).to(Args.device)

    logger.print(encoder, file="models/encoder.txt")
    logger.print(decoder, file="models/decoder.txt")

    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    optimizer = getattr(torch.optim, Args.optimizer)([
        *encoder.parameters(), *decoder.parameters()], Args.lr)

    from plan2vec.plan2vec.maze_plan2vec import sample_trajs

    trajs = sample_trajs(seed=Args.seed, env_id=Args.env_id, num_rollouts=Args.num_rollouts, obs_keys=Args.obs_keys,
                         limit=Args.limit)

    all_images = torch.tensor(np.concatenate([p[Args.image_key] for p in trajs]), requires_grad=False).float().to(
        Args.device)

    shuffled = all_images[torch.randperm(len(all_images))]

    if Args.k_fold:
        N = len(all_images)
        train_data = shuffled[N // Args.k_fold:]
        test_data = shuffled[:N // Args.k_fold]
    else:
        train_data = shuffled
        test_data = shuffled

    train_loader = StackLoader(train_data, Args.batch_size, Args.device)
    test_loader = StackLoader(test_data, Args.batch_size, Args.device)

    def generate_images():
        with torch.no_grad():
            for xs in test_loader:
                _ = encoder(xs)
                mu, logvar = _[:, :Args.latent_dim], _[:, Args.latent_dim:]
                sampled_c = reparameterize(mu, logvar)
                x_bars = decoder(sampled_c)
                break

            _ = torch.cat([*xs[0:10], *x_bars[0:10], *xs[10:20], *x_bars[10:20], ])
            _ = (_ * 255).cpu().numpy().astype('uint8')
            logger.log_images(_[:, :, :, None], f"figures/vae_{epoch:04d}.png", n_rows=4, n_cols=10)
        return _

    for epoch in range(Args.num_epochs):
        for xs in tqdm(train_loader):
            _ = encoder(xs)
            mu, logvar = _[:, :Args.latent_dim], _[:, Args.latent_dim:]
            sampled_c = reparameterize(mu, logvar)

            x_bars = decoder(sampled_c)
            B, *_ = x_bars.shape
            loss = F.binary_cross_entropy(x_bars.view(B, -1), xs.view(B, -1), reduction='sum')
            loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            logger.store_metrics(loss=loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        generate_images()
        logger.log_metrics_summary(loss="min_max", key_values=dict(
            epoch=epoch, dt_epoch=logger.split()
        ))

    logger.save_module(encoder, path="models/encoder.pkl", chunk=200_000_000)
    logger.save_module(decoder, path="models/decoder.pkl", chunk=200_000_000)

    print('training is complete')


if __name__ == "__main__":
    train()
