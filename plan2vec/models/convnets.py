import torch
from torch import nn as nn
from torch_utils import View, Λ

class LocalMetricConv(nn.Module):
    def __init__(self, input_dim, *_, **__):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(input_dim * 2, 16, kernel_size=4, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            View(128 * 36),
            nn.Linear(128 * 36, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        _ = torch.cat([x, x_prime], dim=-3)
        *b, C, H, W = _.shape
        _ = self.trunk(_.reshape(-1, C, H, W))
        return _.reshape(*b, 1)


class LocalMetricConvL2(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=4, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            View(128 * 36),
            nn.Linear(128 * 36, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.kernel = Λ(lambda a, b: (a - b).norm(2, dim=-1))

    def forward(self, x, x_prime):
        *b, C, H, W = x.shape
        *b_, C, H, W = x_prime.shape
        z_1, z_2 = torch.broadcast_tensors(
            self.embed(x.reshape(-1, C, H, W)).reshape(*b, self.latent_dim),
            self.embed(x_prime.reshape(-1, C, H, W)).reshape(*b_, self.latent_dim))
        *b, W = z_1.shape
        return self.kernel(z_1, z_2).reshape(*b, 1)


class LocalMetricCoordConv(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        import numpy as np
        _ = np.linspace(0, 1, 65)[1:]
        _ = np.stack(np.meshgrid(_, _))

        self.register_buffer(
            'coord', torch.tensor(_, requires_grad=False, dtype=torch.float32)[None, ...])
        self.trunk = nn.Sequential(
            nn.Conv2d(input_dim * 2 + 2, 16, kernel_size=4, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            View(128 * 36),
            nn.Linear(128 * 36, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        _ = torch.cat([x, x_prime, self.coord.repeat([*x.shape[:-3], 1, 1, 1])], dim=-3)
        *b, C, H, W = _.shape
        _ = self.trunk(_.reshape(-1, C, H, W))
        return _.reshape(*b, 1)


class LocalMetricConvLarge(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(input_dim * 2, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            View(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        _ = torch.cat([x, x_prime], dim=-3)
        *b, C, H, W = _.shape
        return self.trunk(_.reshape(-1, C, H, W)).reshape(*b, 1)


class LocalMetricConvLargeKernel(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.trunk = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            View(128),
            nn.Linear(128, latent_dim),
        )
        self.kernel = nn.Sequential(
            nn.Linear(latent_dim * 2, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )

    def forward(self, x, x_prime):
        *b, C, H, W = x.shape
        *b_, C, H, W = x_prime.shape
        z_1, z_2 = torch.broadcast_tensors(self.trunk(x.reshape(-1, C, H, W)).reshape(*b, self.latent_dim),
                                           self.trunk(x_prime.reshape(-1, C, H, W)).reshape(*b_, self.latent_dim))
        _ = torch.cat([z_1, z_2], dim=-1)
        *b, W = _.shape
        return self.kernel(_.reshape(-1, W)).reshape(*b, 1)


class LocalMetricConvLargeL2(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            View(128),
            nn.Linear(128, latent_dim),
        )
        self.kernel = Λ(lambda a, b: (a - b).norm(2, dim=-1))

    def forward(self, x, x_prime):
        *b, C, H, W = x.shape
        *b_, C, H, W = x_prime.shape
        z_1, z_2 = torch.broadcast_tensors(
            self.embed(x.reshape(-1, C, H, W)).reshape(*b, self.latent_dim),
            self.embed(x_prime.reshape(-1, C, H, W)).reshape(*b_, self.latent_dim))
        *b, W = z_1.shape
        return self.kernel(z_1, z_2).reshape(*b, 1)


class LocalMetricConvXL(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(input_dim * 2, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            View(128 * 4),
            nn.Linear(128 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        _ = torch.cat([x, x_prime], dim=-3)
        *b, C, H, W = _.shape
        return self.trunk(_.reshape(-1, C, H, W)).reshape(*b, 1)


class LocalMetricConvDeep(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(input_dim * 2, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            View(128 * 4),
            nn.Linear(128 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        _ = torch.cat([x, x_prime], dim=-3)
        *b, C, H, W = _.shape
        return self.trunk(_.reshape(-1, C, H, W)).reshape(*b, 1)


class LocalMetricRGBWide(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(input_dim * 2, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            View(128 * 4),
            nn.Linear(128 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        _ = torch.cat([x, x_prime], dim=-3)
        *b, C, H, W = _.shape
        return self.trunk(_.reshape(-1, C, H, W)).reshape(*b, 1)


class GlobalMetricConvDeepKernel(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 10, kernel_size=4, stride=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            View(40),
            nn.Linear(40, latent_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim * 2, 1),
        )

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        *b, C, H, W = x.shape
        _ = torch.cat([self.embed(x.reshape(-1, C, H, W)),
                       self.embed(x_prime.reshape(-1, C, H, W))], dim=-1)
        return self.head(_).reshape(*b, 1)


class GlobalMetricConvDeepL2(nn.Module):
    def __init__(self, input_dim, latent_dim, p=2):
        super().__init__()

        self.p = 2
        self.embed = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            View(128 * 4),
            nn.Linear(128 * 4, latent_dim),
        )
        self.head = Λ(lambda a, b: (a - b).norm(p, dim=-1))

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        *b, C, H, W = x.shape
        _ = self.embed(x.reshape(-1, C, H, W)), self.embed(x_prime.reshape(-1, C, H, W))
        return self.head(*_).reshape(*b, 1)


class GlobalMetricConvDeepL2_big_head(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            View(128 * 4),
            nn.Linear(128 * 4, latent_dim)
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim * 2, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        *b, C, H, W = x.shape
        _ = self.embed(x.reshape(-1, C, H, W)), self.embed(x_prime.reshape(-1, C, H, W))
        return self.head(*_).reshape(*b, 1)


class GlobalMetricConvDeepL2_ablation(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 10, kernel_size=4, stride=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            View(10 * 4),
            nn.Linear(10 * 4, latent_dim),
        )
        self.head = Λ(lambda a, b: (a - b).norm(2, dim=-1))

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        *b, C, H, W = x.shape
        _ = self.embed(x.reshape(-1, C, H, W)), self.embed(x_prime.reshape(-1, C, H, W))
        return self.head(*_).reshape(*b, 1)


class GlobalMetricConvDeepL2_wide(nn.Module):
    def __init__(self, input_dim, latent_dim, p=2):
        super().__init__()

        self.p = 2
        self.embed = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            View(256 * 4),
            nn.Linear(256 * 4, latent_dim),
        )
        self.head = Λ(lambda a, b: (a - b).norm(p, dim=-1))

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        *b, C, H, W = x.shape
        _ = self.embed(x.reshape(-1, C, H, W)), self.embed(x_prime.reshape(-1, C, H, W))
        return self.head(*_).reshape(*b, 1)


class GlobalMetricConvL2_s1(nn.Module):
    """ stride is 1. """

    def __init__(self, input_dim, latent_dim, p=2):
        super().__init__()

        self.p = 2
        self.embed = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=7, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=7, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=7, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=7, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=7, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=2, stride=1),
            nn.ReLU(),
            View(256),
            nn.Linear(256, latent_dim),
        )
        self.head = Λ(lambda a, b: (a - b).norm(p, dim=-1))

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        *b, C, H, W = x.shape
        _ = self.embed(x.reshape(-1, C, H, W)), self.embed(x_prime.reshape(-1, C, H, W))
        return self.head(*_).reshape(*b, 1)


class GlobalMetricCoordConvL2(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        import numpy as np
        _ = np.linspace(0, 1, 65)[1:]
        _ = np.stack(np.meshgrid(_, _))

        self.register_buffer(
            'coord', torch.tensor(_, requires_grad=False, dtype=torch.float32)[None, ...])

        self.trunk = nn.Sequential(
            nn.Conv2d(input_dim + 2, 128, kernel_size=7, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=7, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=7, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=7, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=7, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=2, stride=1),
            nn.ReLU(),
            View(256),
            nn.Linear(256, latent_dim),
        )
        self.head = Λ(lambda a, b: (a - b).norm(2, dim=-1))

    def embed(self, x):
        _ = torch.cat([x, self.coord.repeat([*x.shape[:-3], 1, 1, 1])], dim=-3)
        return self.trunk(_)

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        *b, C, H, W = x.shape
        _ = self.embed(x.reshape(-1, C, H, W)), self.embed(x_prime.reshape(-1, C, H, W))
        return self.head(*_).reshape(*b, 1)


class GlobalMetricConvDeepL1(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=4, stride=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(),
            View(latent_dim * 4),
            nn.Linear(latent_dim * 4, latent_dim)
        )

    def encode(self, x):
        return self.trunk(x)

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        *b, C, H, W = x.shape
        z = self.trunk(x.reshape(-1, C, H, W))
        z_prime = self.trunk(x_prime.reshape(-1, C, H, W))
        return torch.sum(torch.abs(z_prime - z), dim=-1)  # L1 distance


def get(name):
    return eval(name)


if __name__ == "__main__":
    from termcolor import cprint

    cprint("running tests", "green")
    image = torch.ones(1, 1, 64, 64)
    model = GlobalMetricConvL2_s1(None, 2)
    out = model.embed(image)
    print(out.shape)
    out = model(image, image)
    print(out.shape)

    image = torch.ones(10, 1, 64, 64)
    model = GlobalMetricConvL2_s1(None, 2)
    out = model.embed(image)
    print(out.shape)
    out = model(image, image)
    print(out.shape)

    cprint("running tests", "green")
    image = torch.ones(1, 1, 64, 64)
    model = GlobalMetricCoordConvL2(None, 2)
    out = model.embed(image)
    print(out.shape)
    out = model(image, image)
    print(out.shape)
