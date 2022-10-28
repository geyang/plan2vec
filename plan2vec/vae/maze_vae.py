from params_proto import cli_parse
import torch
from torch import nn
from tqdm import tqdm

from env_data.batch_loader import BatchLoader


@cli_parse
class Args:
    env_name = "PointMass-v0"
    act_dim = 2
    obs_dim = 8

    latent_dim = 10

    batch_size = 100
    test_batch_size = 2
    n_workers = 5

    n_epochs = 400
    lr = 1e-4  # 0.01 for SGD
    optimizer = "Adam"

    use_gpu = True

    debug_interval = 10


def train(_Args=None):
    Args.update(_Args or {})
    import torch.nn.functional as F
    from torch_utils import View
    from ml_logger import logger

    logger.log_params(Args=vars(Args))

    device = torch.device("cuda" if Args.use_gpu and torch.cuda.is_available() else "cpu")

    encoder = nn.Sequential(
        # 28 x 28
        nn.Conv2d(3, 32, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=5, stride=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=6, stride=1),
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=4, stride=1),
        nn.ReLU(),
        View(-1),
        nn.Linear(256, Args.latent_dim * 2),
    )

    decoder = nn.Sequential(
        View(Args.latent_dim, 1, 1),
        nn.ConvTranspose2d(Args.latent_dim, 128, kernel_size=4, stride=1),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, kernel_size=6, stride=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 3, kernel_size=7, stride=1),
        nn.Sigmoid(),
    )

    state_decoder = nn.Sequential(
        # todo: play with more layers
        nn.Linear(Args.latent_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, Args.obs_dim),
    )

    two_frame_state_decoder = nn.Sequential(
        nn.Linear(Args.latent_dim * 2 + Args.act_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, Args.obs_dim),
    )

    logger.log_params(
        decoder=str(decoder),
        encoder=str(encoder),
        state_decoder=str(state_decoder),
    )

    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    encoder.to(device)
    decoder.to(device)
    state_decoder.to(device)
    two_frame_state_decoder.to(device)

    optimizer = getattr(torch.optim, Args.optimizer)([*encoder.parameters(), *decoder.parameters()], Args.lr)
    state_optimizer = getattr(torch.optim, Args.optimizer)(state_decoder.parameters(), Args.lr)
    two_frame_state_optimizer = getattr(torch.optim, Args.optimizer)(two_frame_state_decoder.parameters(), Args.lr)

    path = f'./data/{Args.env_name}/samples.pkl'
    train_loader = BatchLoader(path, Args.batch_size, shuffle="timestep", device=device)
    # todo-2: use different data for test and train
    test_path = f'./data/{Args.env_name}/validate/samples.pkl'
    test_loader = BatchLoader(test_path, Args.test_batch_size, shuffle="rollout", device=device)

    pair = lambda xs, dim: torch.cat([xs[:-1], xs[1:]], dim=dim)

    def save_debug_samples():
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                s, a, img = data['obs'], data['acs'], data['views']
                timesteps, batch_size = img.shape[:2]
                xs = img.permute(0, 1, 4, 2, 3) / 255
                _ = encoder(xs.reshape(-1, 3, 28, 28))
                mu, logvar = _[:, :Args.latent_dim], _[:, Args.latent_dim:]

                s_bars = state_decoder(mu).reshape(s.shape)  # train with only the mu. s ind offset is important
                mu_pairs = pair(mu.reshape([*s.shape[:-1], Args.latent_dim]), dim=-1)
                mu_act_pairs = torch.cat([mu_pairs, a[:-1]], dim=-1)
                two_frame_s_bars = two_frame_state_decoder(mu_act_pairs).reshape(s[1:].shape)  # train with only the mu.

                state_loss = F.mse_loss(s_bars, s)
                theta_loss = F.mse_loss(s_bars[:, :, :2], s[:, :, :2], reduction="mean")
                theta_dot_loss = F.mse_loss(s_bars[:, :, 2:], s[:, :, 2:], reduction="mean")

                two_frame_state_loss = F.mse_loss(two_frame_s_bars, s[1:])
                two_frame_theta_loss = F.mse_loss(two_frame_s_bars[:, :, :2], s[1:, :, :2], reduction="mean")
                two_frame_theta_dot_loss = F.mse_loss(two_frame_s_bars[:, :, 2:], s[1:, :, 2:], reduction="mean")

                # report all norms + MSE of prediction
                logger.store_metrics(metrics={
                    "test/state_mse": state_loss.item(),
                    "test/theta_mse": theta_loss.item(),
                    "test/theta_dot_mse": theta_dot_loss.item(),

                    "test/two_frame_state_mse": two_frame_state_loss.item(),
                    "test/two_frame_theta_mse": two_frame_theta_loss.item(),
                    "test/two_frame_theta_dot_mse": two_frame_theta_dot_loss.item(),
                })

                if i == 0:
                    x_bars = decoder(mu).reshape(xs.shape)
                    # todo-1: generate debug samples
                    # todo-2: report state prediction error here
                    _ = torch.cat([xs, x_bars], dim=-2).permute(1, 0, 3, 4, 2)
                    _ = (_ * 255).cpu().numpy().reshape(-1, 56, 28, 3).astype('uint8')
                    logger.log_images(_, f"samples/vae_{epoch:04d}.png", n_rows=batch_size, n_cols=timesteps)
                    logger.log_video(_, f"samples/vae_{epoch:04d}.gif", fps=6)

    for epoch in range(Args.n_epochs + 1):
        if epoch % Args.debug_interval == 0:
            save_debug_samples()

        for data in tqdm(train_loader):
            s, a, img = data['obs'], data['acs'], data['views']
            with torch.no_grad():
                xs = img.permute(0, 1, 4, 2, 3) / 255

            _ = encoder(xs.reshape(-1, 3, 28, 28))
            mu, logvar = _[:, :Args.latent_dim], _[:, Args.latent_dim:]
            sampled_c = reparameterize(mu, logvar)
            x_bars = decoder(sampled_c)
            # use reshape because incongruous.
            decoder_loss = F.binary_cross_entropy(x_bars.reshape(-1, 28 * 28 * 3), xs.reshape(-1, 28 * 28 * 3),
                                                  reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = kl_loss + decoder_loss

            logger.store_metrics(decoder_loss=decoder_loss.item(), kl_loss=kl_loss.item(), total_loss=loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # todo-3: add state space prediction
            # todo-5: use sampled c for training.
            s_bars = state_decoder(mu.detach()).reshape(s.shape)  # train with only the mu.
            with torch.no_grad():
                # mu_pairs = pair(mu.reshape([*s.shape[:-1], Args.latent_dim]), dim=-1)
                c_pairs = pair(sampled_c.reshape([*s.shape[:-1], Args.latent_dim]), dim=-1)
                c_act_pairs = torch.cat([c_pairs, a[:-1]], dim=-1)
            # todo-6: add action to see if dot can be predicted.
            two_frame_s_bars = two_frame_state_decoder(c_act_pairs).reshape(s[1:].shape)  # train with only the mu.
            # todo-4: per component
            state_loss = F.mse_loss(s_bars, s)
            two_frame_state_loss = F.mse_loss(two_frame_s_bars, s[1:])

            theta_loss = F.mse_loss(s_bars[:, :, :2], s[:, :, :2], reduction="mean")
            theta_dot_loss = F.mse_loss(s_bars[:, :, 2:], s[:, :, 2:], reduction="mean")

            two_frame_theta_loss = F.mse_loss(two_frame_s_bars[:, :, :2], s[1:, :, :2], reduction="mean")
            two_frame_theta_dot_loss = F.mse_loss(two_frame_s_bars[:, :, 2:], s[1:, :, 2:], reduction="mean")

            with torch.no_grad():
                theta_norm = torch.sqrt((s[:, :, :2] ** 2).mean(dim=-1))
                theta_dot_norm = torch.sqrt((s[:, :, 2:] ** 2).mean(dim=-1))

            # report all norms + MSE of prediction
            logger.store_metrics(state_mse=state_loss.item(),
                                 theta_mse=theta_loss.item(),
                                 theta_dot_mse=theta_dot_loss.item(),

                                 two_frame_state_mse=two_frame_state_loss.item(),
                                 two_frame_theta_mse=two_frame_theta_loss.item(),
                                 two_frame_theta_dot_mse=two_frame_theta_dot_loss.item(),

                                 theta_norm=theta_norm.cpu().numpy(),
                                 theta_dot_norm=theta_dot_norm.cpu().numpy(),
                                 )

            state_loss.backward()
            state_optimizer.step()
            state_optimizer.zero_grad()

            two_frame_state_loss.backward()
            two_frame_state_optimizer.step()
            two_frame_state_optimizer.zero_grad()
            # todo-4: add frame-difference prediction (img + act -> img_delta)

        if epoch % 20 == 0:
            # use save_modules is faster, but this way we can load just one module.
            logger.log_line(f'saving modules @ epoch {epoch}...')
            logger.save_module(encoder, path=f"weights/{epoch:04d}_encoder.pkl")
            logger.save_module(decoder, path=f"weights/{epoch:04d}_decoder.pkl")

        logger.log_metrics_summary(
            key_values=dict(epoch=epoch, dt_epoch=logger.split()))

    print('training is complete')


if __name__ == "__main__":
    train()
