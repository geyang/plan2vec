from os.path import expanduser

from params_proto import cli_parse
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@cli_parse
class Args:
    batch_size = 100
    n_workers = 5
    pin_memory = True

    download_mnist = False

    num_epochs = 100
    lr = 1e-4  # 0.01 for SGD
    optimizer = "Adam"

    use_gpu = True


def train(**kwargs):
    Args.update(**kwargs)
    from jaynes.helpers import pick
    from torchvision import datasets, transforms
    import torch.nn.functional as F
    from torch_utils import View
    from ml_logger import logger

    logger.log_params(Args=vars(Args))

    device = torch.device("cuda" if Args.use_gpu and torch.cuda.is_available() else "cpu")

    encoder = nn.Sequential(
        # 28 x 28
        nn.Conv2d(1, 32, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=5, stride=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=6, stride=1),
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=4, stride=1),
        nn.ReLU(),
        View(256),
        nn.Linear(256, 10 + 10),
    )

    decoder = nn.Sequential(
        View(10, 1, 1),
        nn.ConvTranspose2d(10, 128, kernel_size=4, stride=1),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, kernel_size=6, stride=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 1, kernel_size=7, stride=1),
        nn.Sigmoid(),
    )

    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    encoder.to(device)
    decoder.to(device)

    optimizer = getattr(torch.optim, Args.optimizer)([
        *encoder.parameters(), *decoder.parameters()], Args.lr)

    train_loader = DataLoader(
        datasets.MNIST(expanduser('~/data'), train=True, download=Args.download_mnist,
                       transform=transforms.ToTensor()),
        batch_size=Args.batch_size, shuffle=True, **pick(vars(Args), "num_workers", "pin_memory"))
    test_loader = DataLoader(
        datasets.MNIST(expanduser('~/data'), train=False, transform=transforms.ToTensor()),
        batch_size=Args.batch_size, shuffle=True, **pick(vars(Args), "num_workers", "pin_memory"))

    def generate_images():
        with torch.no_grad():
            for xs, _ in test_loader:
                xs = xs.to(device)
                _ = encoder(xs)
                mu, logvar = _[:, :10], _[:, 10:]
                sampled_c = reparameterize(mu, logvar)
                x_bars = decoder(sampled_c)
                break

            _ = torch.cat([*xs[0:10], *x_bars[0:10], *xs[10:20], *x_bars[10:20], ])
            _ = (_ * 255).cpu().numpy().astype('uint8')
            logger.log_images(_[:, :, :, None], f"figures/vae_{epoch:04d}.png", n_rows=4, n_cols=10)
        return _

    for epoch in range(Args.num_epochs):
        for xs, labels in tqdm(train_loader):
            xs = xs.to(device)

            _ = encoder(xs)
            mu, logvar = _[:, :10], _[:, 10:]
            sampled_c = reparameterize(mu, logvar)
            x_bars = decoder(sampled_c)
            loss = F.binary_cross_entropy(x_bars.view(-1, 784), xs.view(-1, 784), reduction='sum')
            loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            logger.store_metrics(loss=loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        generate_images()
        logger.log_metrics_summary(loss="min_max", key_values=dict(
            epoch=epoch, dt_epoch=logger.split()
        ))

    print('training is complete')


if __name__ == "__main__":
    train()
