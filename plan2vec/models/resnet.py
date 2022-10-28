import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.resnet as resnet
import numpy as np

from torch_utils import Λ


class ResNet18(models.ResNet):
    """
    for documentation see

        https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnet18
    """

    def __init__(self, input_dim, num_classes, is_gaussian=False):
        block = resnet.BasicBlock
        super().__init__(block, [2, 2, 2, 2], num_classes)
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if is_gaussian:  # handle variational output
            self.var = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def gaussian(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return self.fc(x), self.var(x)


class ResNet18Coord(models.ResNet):
    """
    for documentation see

        https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnet18
    """

    def __init__(self, input_dim, num_classes):
        super().__init__(resnet.BasicBlock, [2, 2, 2, 2], num_classes)
        self.conv1 = nn.Conv2d(input_dim + 2, 64, kernel_size=7, stride=2, padding=3, bias=False)

        import numpy as np
        _ = np.linspace(0, 1, 65)[1:]
        _ = np.stack(np.meshgrid(_, _))

        self.register_buffer(
            'coord', torch.tensor(_, requires_grad=False, dtype=torch.float32)[None, ...])

    def forward(self, x):
        _ = torch.cat([x, self.coord.repeat([*x.shape[:-3], 1, 1, 1])], dim=-3)
        return super().forward(_)


class ResNet18L2(nn.Module):
    def __init__(self, input_dim, latent_dim, p=2):
        super().__init__()
        self.p = p
        self.embed = ResNet18(input_dim, latent_dim)
        self.head = Λ(lambda a, b: (a - b).norm(p, dim=-1))

    def forward(self, x, x_prime):
        # note: for some reason, if x_prime has batch size of 1, it works poorly with DataParallel.
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        *b, C, H, W = x.shape
        _ = self.embed(x.reshape(-1, C, H, W)), self.embed(x_prime.reshape(-1, C, H, W))
        return self.head(*_).reshape(*b, 1)


class ResNet18AsymmetricL2(nn.Module):
    def __init__(self, input_dim, latent_dim, p=2):
        super().__init__()
        self.p = p
        self.embed = ResNet18(input_dim, latent_dim)
        self.embed_2 = ResNet18(input_dim, latent_dim)
        self.head = Λ(lambda a, b: (a - b).norm(p, dim=-1))

    def forward(self, x, x_prime):
        # note: for some reason, if x_prime has batch size of 1, it works poorly with DataParallel.
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        *b, C, H, W = x.shape
        _ = self.embed(x.reshape(-1, C, H, W)), self.embed_2(x_prime.reshape(-1, C, H, W))
        return self.head(*_).reshape(*b, 1)


class ResNet18AsymmetricKernel(nn.Module):
    def __init__(self, input_dim, latent_dim, p=2):
        super().__init__()
        from ml_logger import logger
        logger.upload_file(__file__)
        self.p = p
        self.embed = ResNet18(input_dim, latent_dim)
        # self.embed_2 = ResNet18(input_dim, latent_dim)
        self.embed_2 = self.embed
        self.kernel = nn.Sequential(
            nn.Linear(latent_dim * 2, 400),
            nn.Sigmoid(),
            nn.Linear(400, 400),
            nn.Sigmoid(),
            nn.Linear(400, 400),
            nn.Sigmoid(),
            nn.Linear(400, 20),
            nn.Sigmoid(),
            nn.Linear(20, 1),
        )
        self.head = Λ(lambda a, b: self.kernel(torch.cat([a, b], -1)))

    def forward(self, x, x_prime):
        # note: for some reason, if x_prime has batch size of 1, it works poorly with DataParallel.
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        *b, C, H, W = x.shape
        _ = self.embed(x.reshape(-1, C, H, W)), self.embed_2(x_prime.reshape(-1, C, H, W))
        return self.head(*_).reshape(*b, 1)


class ResNet18CoordL2(nn.Module):
    def __init__(self, input_dim, latent_dim, p=2):
        super().__init__()
        self.input_dim = input_dim  # used for shape inference by other modules
        self.latent_dim = latent_dim  # used for shape inference by other modules
        self.embed = ResNet18Coord(input_dim, latent_dim)
        self.head = Λ(lambda a, b: (a - b).norm(p, dim=-1))

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        *b, C, H, W = x.shape
        _ = self.embed(x.reshape(-1, C, H, W)), self.embed(x_prime.reshape(-1, C, H, W))
        return self.head(*_).reshape(*b, 1)


class ResNet18CoordAsymmetricL2(nn.Module):
    """Uses two separate trunks without weight tying"""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.embed = ResNet18Coord(input_dim, latent_dim)
        self.embed_2 = ResNet18Coord(input_dim, latent_dim)
        self.head = Λ(lambda a, b: (a - b).norm(2, dim=-1))

    def forward(self, x, x_prime):
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        *b, C, H, W = x.shape
        _ = self.embed(x.reshape(-1, C, H, W)), self.embed_2(x_prime.reshape(-1, C, H, W))
        return self.head(*_).reshape(*b, 1)


class ResNet18Kernel(nn.Module):
    """The expressivity of this 1-layer linear kernel is limited."""

    def __init__(self, input_dim, latent_dim, **_):
        super().__init__()
        self.embed = ResNet18(input_dim, latent_dim)
        self.kernel = nn.Sequential(
            nn.Linear(latent_dim * 2, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
        self.head = Λ(lambda a, b: self.kernel(torch.cat([a, a - b], -1)))

    def forward(self, x, x_prime):
        # note: for some reason, if x_prime has batch size of 1, it works poorly with DataParallel.
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        *b, C, H, W = x.shape
        _ = self.embed(x.reshape(-1, C, H, W)), self.embed(x_prime.reshape(-1, C, H, W))
        return self.head(*_).reshape(*b, 1)


class ResNet18Stacked(nn.Module):
    """Stacked ResNet18, with C + C input channels."""

    def __init__(self, input_dim, latent_dim=10):
        super().__init__()
        self.embed = ResNet18(input_dim * 2, latent_dim)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, x, x_prime):
        # note: for some reason, if x_prime has batch size of 1, it works poorly with DataParallel.
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        *b, C, H, W = x.shape
        stack = x.reshape(-1, C, H, W), x_prime.reshape(-1, C, H, W)
        _ = self.embed(torch.cat(stack, dim=-3))
        return self.head(_).reshape(*b, 1)


if __name__ == "__main__":
    samples = np.stack([np.ones([3, 64, 64]), np.zeros([3, 64, 64])])

    # model = models.resnet18()
    model = ResNet18(3, 2)
    x, y = torch.tensor(samples, dtype=torch.float32), torch.tensor([0, 1], dtype=torch.float32)
    y_hat = model(x)
    print(y_hat.shape)

    from termcolor import cprint

    cprint("running tests", "green")

    image = torch.ones(1, 3, 64, 64)

    model = ResNet18CoordL2(3, 2)
    out = model.embed(image)
    print(out.shape)
    out = model(image, image)
    print(out.shape)

    model = ResNet18Stacked(3)
    out = model(image, image)
    print(out.shape)
