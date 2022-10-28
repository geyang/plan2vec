import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.resnet as resnet
import numpy as np


class ResNet18Gray(models.ResNet):
    def __init__(self):
        super().__init__(resnet.BasicBlock, [2, 2, 2, 2], num_classes=1)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)


if __name__ == "__main__":
    samples = np.stack([np.ones([1, 64, 64]), np.zeros([1, 64, 64])])

    # model = models.resnet18()
    model = ResNet18Gray()

    x, y = torch.tensor(samples, dtype=torch.float32), torch.tensor([0, 1], dtype=torch.float32)

    y_hat = model(x)
    y_hat.shape

