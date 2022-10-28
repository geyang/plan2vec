import torch
from torch import nn as nn

from torch_utils import Λ


class GlobalMetricFusion(nn.Module):
    """
    This one did not work, but the larger (and deeper) model worked.
    """

    def __init__(self, input_dim, latent_dim=50):
        super().__init__()
        self.trunk = nn.Linear(input_dim, latent_dim)
        self.head = nn.Sequential(
            nn.Linear(latent_dim * 2, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    embed = None

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        *b, W = x.shape
        _ = torch.cat([self.trunk(x.reshape(-1, W)), self.trunk(x_prime.reshape(-1, W))], dim=-1)
        return self.head(_).reshape(*b, 1)


class GlobalMetricMlp(nn.Module):
    """
    This one did not work, but the larger (and deeper) model worked.
    """

    def __init__(self, input_dim, latent_dim=50):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1)
        )

    embed = None

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        *b, W = x.shape
        _ = torch.cat([x.reshape(-1, W), x_prime.reshape(-1, W)], dim=-1)
        return self.model(_).reshape(*b, 1)


class GlobalMetricLinearKernel(nn.Module):
    """
    This one worked.
    """

    def __init__(self, input_dim, latent_dim=50):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, latent_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim * 2, 1),
        )

    def encode(self, x):
        return self.embed(x)

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        *b, W = x.shape
        _ = torch.cat([self.embed(x.reshape(-1, W)), self.embed(x_prime.reshape(-1, W))], dim=-1)
        return self.head(_).reshape(*b, 1)


class GlobalMetricKernel(nn.Module):
    """
    This one worked.
    """

    def __init__(self, input_dim, latent_dim=50):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, latent_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim * 2, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )

    def encode(self, x):
        return self.embed(x)

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        *b, W = x.shape
        _ = torch.cat([self.embed(x.reshape(-1, W)), self.embed(x_prime.reshape(-1, W))], dim=-1)
        return self.head(_).reshape(*b, 1)


class GlobalMetricL2(nn.Module):
    """
    This one worked better, because of the constraint as an L2 metric.
    """

    def __init__(self, input_dim, latent_dim=50):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, latent_dim),
        )
        self.head = Λ(lambda a, b: (a - b).norm(2, dim=-1))

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        *b, W = x.shape
        _ = self.embed(x.reshape(-1, W)), self.embed(x_prime.reshape(-1, W))
        return self.head(*_).reshape(*b, 1)


class GlobalMetricAsymmetricL2(nn.Module):
    """
    Instead of using an L2, we can use asymmetric trunks without weight tying.
    """

    def __init__(self, input_dim, latent_dim=50):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, latent_dim),
        )
        self.embed_2 = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, latent_dim),
        )
        self.head = Λ(lambda a, b: (a - b).norm(2, dim=-1))

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        *b, W = x.shape
        _ = self.embed(x.reshape(-1, W)), self.embed_2(x_prime.reshape(-1, W))
        return self.head(*_).reshape(*b, 1)


class LocalMetric(nn.Module):
    """Returns 1 if inputs are neighbors, 0 otherwise
    """

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.embed = nn.Linear(input_dim, latent_dim)
        self.input_dim = input_dim
        self.head = nn.Sequential(
            nn.Linear(latent_dim * 2, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        *b, W = x.shape
        _ = torch.cat([self.embed(x.reshape(-1, W)), self.embed(x_prime.reshape(-1, W))], dim=-1)
        return self.head(_).reshape(*b, 1)
