from os.path import expanduser

from params_proto import cli_parse, Proto
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from plan2vec.plotting.streetlearn.embedding_image_streetlearn import \
    cache_images, \
    visualize_embedding_2d_image, \
    visualize_embedding_3d_image


@cli_parse
class Args:
    seed = 0

    # environment specs
    env_id = 'manhattan-tiny'

    data_path = Proto(f"~/fair/streetlearn/processed-data/{env_id}", help="path to the processed streetlearn dataset")
    street_view_size = Proto((64, 64), help="image size for the dataset", dtype=tuple)
    street_view_mode = Proto("omni-gray", help="OneOf[`omni-gray`, `ombi-rgb`]")

    lng_lat_correction = Proto(0.75, help="length correction factor for lattitude. Different by lattitude.")

    latent_dim = 2

    k_fold = 10

    batch_size = 100
    n_workers = 5

    download_mnist = False

    n_epochs = 5000
    lr = 1e-4  # 0.01 for SGD
    optimizer = "Adam"

    use_gpu = True

    checkpoint_last = True


class SingleDataset(Dataset):
    def __init__(self, data, dtype=torch.float, device=None):
        self.data = data
        self.dtype = dtype
        self.device = device

    def __getitem__(self, index):
        import numpy as np
        if isinstance(index, int):
            return torch.tensor(self.data[index], dtype=self.dtype, device=self.device)
        elif isinstance(index, slice):
            data_len = len(self.data)
            # this can potentially introduce a memory leak.
            start = np.floor(data_len * index.start).astype(int) if isinstance(index.start, float) else index.start
            stop = np.floor(data_len * index.stop).astype(int) if isinstance(index.stop, float) else index.stop
            return SingleDataset(data=None if self.data is None else self.data[start:stop],
                                 device=self.device)
        else:
            raise KeyError(f"{index} slice is not supported!")

    def __len__(self):
        return len(self.data)


def train(**kwargs):
    import numpy as np
    import torch.nn.functional as F
    from torch_utils import View
    from ml_logger import logger

    Args.update(**kwargs)
    device = torch.device("cuda" if Args.use_gpu and torch.cuda.is_available() else "cpu")
    logger.log_params(Args=vars(Args))

    np.random.seed(Args.seed)
    torch.manual_seed(Args.seed)
    # see https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from plan2vec.models.resnet import ResNet18

    # use 2x of the latent_dim for variance
    encoder = ResNet18(1, Args.latent_dim, is_gaussian=True)

    decoder = nn.Sequential(
        View(Args.latent_dim, 1, 1),
        nn.ConvTranspose2d(Args.latent_dim, 128, kernel_size=4, stride=1),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 128, kernel_size=7, stride=1),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, kernel_size=7, stride=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, kernel_size=7, stride=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, kernel_size=7, stride=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, kernel_size=7, stride=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, kernel_size=7, stride=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=7, stride=1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, kernel_size=7, stride=1),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 8, kernel_size=7, stride=1),
        nn.ReLU(),
        nn.ConvTranspose2d(8, 1, kernel_size=7, stride=1),
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

    from streetlearn import StreetLearnDataset
    streetlearn = StreetLearnDataset(expanduser(Args.data_path), Args.street_view_size, Args.street_view_mode)
    streetlearn.select_all()
    Args.streetlearn_bbox = streetlearn.bbox

    if True:
        from plan2vec.plan2vec.plan2vec_img import pick
        logger.log_params(env=pick(vars(Args), 'streetlearn_bbox', 'env_id', "data_path"))

    dataset = SingleDataset(streetlearn.images[:, None, ...].astype(np.float32) / 255, device=device)
    train_loader = DataLoader(dataset[1 / Args.k_fold:], batch_size=Args.batch_size,
                              shuffle=True, **pick(vars(Args), "num_workers"))
    test_loader = DataLoader(dataset[:1 / Args.k_fold], batch_size=Args.batch_size,
                             shuffle=True, **pick(vars(Args), "num_workers"))

    cache_images(streetlearn.images[:, None, ...].astype(np.float32) / 255, streetlearn.lng_lat,
                 Args.lng_lat_correction)

    def generate_images():
        with torch.no_grad():
            for xs in test_loader:
                # xs = xs.to(device)

                mu, logvar = encoder.gaussian(xs)
                sampled_c = reparameterize(mu.squeeze(-1), logvar.squeeze(-1))

                x_bars = decoder(sampled_c)
                break

            _ = torch.cat([*xs[0:10], *x_bars[0:10], *xs[10:20], *x_bars[10:20], ])
            _ = (_ * 255).cpu().numpy().astype('uint8')
            logger.log_images(_[:, :, :, None], f"figures/samples/vae_{epoch:04d}.png", n_rows=4, n_cols=10)

            if Args.latent_dim in [2, 3]:
                eval(f"visualize_embedding_{Args.latent_dim}d_image")(  # png version for quick view
                    Args.env_id, encoder, f"figures/embedding/embed_{Args.latent_dim}d_{epoch:04d}.png")
        return _

    for epoch in range(Args.n_epochs):
        for xs in tqdm(train_loader):
            # xs = xs.to(device)

            mu, logvar = encoder.gaussian(xs)
            sampled_c = reparameterize(mu.squeeze(-1), logvar.squeeze(-1))
            x_bars = decoder(sampled_c)

            loss = F.binary_cross_entropy(x_bars.view(-1, 64 ** 2), xs.view(-1, 64 ** 2), reduction='sum')
            loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            logger.store_metrics(loss=loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        generate_images()
        logger.log_metrics_summary(loss="min_max", key_values=dict(
            epoch=epoch, dt_epoch=logger.split()
        ))

    if Args.checkpoint_last:
        logger.save_module(encoder, "checkpoints/encoder.pkl", 10_000_000)

    logger.print('training is complete', color="green")


if __name__ == "__main__":
    train()
