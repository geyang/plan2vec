import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils import View, Λ


class Q(nn.Module):
    """the type class for Q-Networks"""
    conv = False

    def __init__(self):
        super().__init__()
        self.softmax = nn.LogSoftmax(-1)

    # note: soft-Q learning is not implemented yet, need to happen
    #  in the training code
    def soft(self, x, g):
        _ = self(x, g)
        return torch.distributions.Categorical(logits=_).sample()

    def hard(self, x, g):
        _ = self(x, g)
        return torch.argmax(_, dim=-1)

    def temperature(self, x, g):
        pass


class QMlp(Q):
    def __init__(self, input_dim, act_dim):
        super().__init__()
        self.trunk = nn.Linear(input_dim, 448)
        self.head = nn.Linear(448, act_dim)

    def forward(self, x, g):
        _ = torch.cat([x, g], dim=-1)
        _ = F.relu(self.trunk(_))
        return self.head(_.view(_.size(0), -1))


class QSharedEmbedding(Q):
    def __init__(self, input_dim, act_dim, latent_dim):
        super().__init__()
        self.embed_fn = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, latent_dim)
        )
        self.kernel = nn.Sequential(
            nn.Linear(latent_dim * 2, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, act_dim)
        )
        self.embed = self.embed_fn

    def forward(self, x, g):
        x = self.embed_fn(x)
        g = self.embed_fn(g)
        _ = torch.cat([x, g], dim=-1)
        return self.kernel(_)


class QL2EmbedModel(Q):
    """This one has a forward model, agnostic to identity action"""

    def __init__(self, input_dim, act_dim, latent_dim, embed_p):
        super().__init__()
        self.embed_p = embed_p
        self.act_dim = act_dim
        self.embed_fn = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, latent_dim),
        )
        self.T = nn.Sequential(
            nn.Linear(latent_dim * 2, 200),
            nn.ReLU(),
            nn.Linear(200, latent_dim)
        )
        self.a_linear = nn.Embedding(act_dim, latent_dim)
        self.embed = self.embed_fn

    def next_embed(self, x, a):
        z_x = self.embed_fn(x)
        a_emb = self.a_linear(a)
        return self.T(torch.cat([z_x, a_emb], dim=-1))

    def forward(self, x, x_g):
        z_x = self.embed_fn(x)
        g = self.embed_fn(x_g).unsqueeze(1)
        _ = torch.arange(self.act_dim).to(x.device)
        a_embs = self.a_linear(_).repeat(z_x.size(0), 1, 1)
        z_xs = z_x.unsqueeze(1).repeat(1, self.act_dim, 1)
        z_x_nexts = self.T(torch.cat([z_xs, a_embs], dim=-1))
        return - torch.norm(z_xs - z_x_nexts, p=self.embed_p, dim=-1) \
               - torch.norm(z_x_nexts - g, p=self.embed_p, dim=-1)


class QL2EmbeddingPassive(Q):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.embed_fn = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, latent_dim)
        )
        self.embed = self.embed_fn

    def forward(self, x, x_, g):
        z_x = self.embed_fn(x)
        z_x_ = self.embed_fn(x_)
        z_g = self.embed_fn(g)
        d_sz = torch.norm(z_x - z_x_, dim=-1)
        d_zg = torch.norm(z_x_ - z_g, dim=-1)
        return - d_sz - d_zg

    def d(self, x, g):
        z_x = self.embed_fn(x)
        z_g = self.embed_fn(g)
        return - torch.norm(z_x - z_g, dim=-1)

    def hop_thy_neighbor(self, x, neighbors, g):
        """
        :param x:
        :param g:
        :param local_metric_fn:
        :return:
        """
        zs = self.embed_fn(neighbors)
        z_x = self.embed_fn(x)[:, None, :]
        z_g = self.embed_fn(g)[:, None, :]
        d_sz = torch.norm(z_x - zs, dim=-1)
        d_zg = torch.norm(zs - z_g, dim=-1)
        q_xg = - d_sz - d_zg
        # change to softmax for thermal sampling
        return q_xg.max(dim=-1)

    max = hop_thy_neighbor


class QL2Embed(Q):
    def __init__(self, input_dim, act_dim, latent_dim, embed_p):
        super().__init__()
        self.embed_p = embed_p
        self.embed_fn = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, latent_dim * act_dim),
            View(act_dim, latent_dim)
        )

    def forward(self, x, x_g):
        z_xs = self.embed_fn(x)
        g = self.embed_fn(x_g)[:, 4, None, :]
        # this one should have two legs.
        return - torch.norm(z_xs[:, 4, None, :] - z_xs, p=self.embed_p, dim=-1) - torch.norm(z_xs - g, p=self.embed_p,
                                                                                             dim=-1)

    def embed(self, x):
        """Only works when identity function is available."""
        # note: maybe we can separate out the embedding function and the identity function? Maybe this is what
        # is breaking the learning down in this case?
        return self.embed_fn(x)[:, 4, :]


class QConv(Q):
    def __init__(self, input_dim, act_dim, latent_dim=None):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            View(128),
            nn.Linear(128, act_dim),
            View(act_dim))

    def forward(self, x, g):
        _ = torch.cat([x, g], dim=1)
        _ = self.trunk(_)
        return _


class QDeepConv(Q):
    def __init__(self, input_dim, act_dim, latent_dim=None):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 5, kernel_size=3, stride=2),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.Conv2d(5, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 10, kernel_size=3, stride=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            View(32),
            nn.Linear(32, act_dim),
            View(act_dim))

    def forward(self, x, g):
        _ = torch.cat([x, g], dim=1)
        _ = self.trunk(_)
        return _


class QL2EmbedConv(Q):
    """this one is stale, requires identity action at [4]"""

    def __init__(self, input_dim, act_dim, latent_dim, embed_p):
        super().__init__()
        self.embed_p = embed_p
        self.embed_fn = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            View(128),
            nn.Linear(128, latent_dim * act_dim),
            View(act_dim, latent_dim))

    def forward(self, x, x_g):
        z_xs = self.embed_fn(x)
        g = self.embed_fn(x_g)[:, 4, None, :]
        # this one should have two legs.
        return - torch.norm(z_xs[:, 4, None, :] - z_xs, p=self.embed_p, dim=-1) \
               - torch.norm(z_xs - g, p=self.embed_p, dim=-1)

    def embed(self, x):
        """Only works when identity function is available."""
        # note: maybe we can separate out the embedding function and the identity function? Maybe this is what
        # is breaking the learning down in this case?
        return self.embed_fn(x)[:, 4, :]


class QSharedEmbeddingConv(Q):
    def __init__(self, input_dim, act_dim, latent_dim):
        super().__init__()
        self.embed_fn = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            View(128),
            nn.Linear(128, latent_dim))
        self.kernel = nn.Sequential(
            nn.Linear(latent_dim * 2, 200),
            nn.ReLU(),
            nn.Linear(200, act_dim)
        )

    def forward(self, x, g):
        x = self.embed_fn(x)
        g = self.embed_fn(g)
        self.z = torch.cat([x, g], dim=-1)
        return self.kernel(self.z)


class QL2EmbeddingPassiveConv(Q):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_fn = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            View(128),
            nn.Linear(128, latent_dim))
        self.embed = self.embed_fn

    def forward(self, x, x_, g):
        z_x = self.embed_fn(x)
        z_x_ = self.embed_fn(x_)
        z_g = self.embed_fn(g)
        d_sz = torch.norm(z_x - z_x_, dim=-1)
        d_zg = torch.norm(z_x_ - z_g, dim=-1)
        return - d_sz - d_zg

    def d(self, x, g):
        z_x = self.embed_fn(x)
        z_g = self.embed_fn(g)
        return - torch.norm(z_x - z_g, dim=-1)

    def hop_thy_neighbor(self, x, neighbors, g):
        """
        :param x:
        :param zs: the latent vectory
        :param g:
        :return:
        """
        num_neighbors = neighbors.shape[1]
        zs = self.embed_fn(neighbors.view(-1, 1, 28, 28)).reshape(-1, num_neighbors, self.latent_dim)
        z_x = self.embed_fn(x)[:, None, :]
        z_g = self.embed_fn(g)[:, None, :]
        d_sz = torch.norm(z_x - zs, dim=-1)
        d_zg = torch.norm(zs - z_g, dim=-1)
        # can't pick same state
        q_xg = - d_sz - d_zg
        # change to softmax for thermal sampling
        return q_xg.max(dim=-1)

    max = hop_thy_neighbor


class QL2EmbedModelConv(Q):
    """This one has a forward model, agnostic to identity action"""

    def __init__(self, input_dim, act_dim, latent_dim, embed_p):
        super().__init__()
        self.embed_p = embed_p
        self.act_dim = act_dim
        self.embed_fn = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            View(128),
            nn.Linear(128, latent_dim))
        self.T = nn.Sequential(
            nn.Linear(latent_dim * 2, 200),
            nn.ReLU(),
            nn.Linear(200, latent_dim)
        )
        self.a_linear = nn.Embedding(act_dim, latent_dim)
        self.embed = self.embed_fn

    def next_embed(self, x, a):
        z_x = self.embed_fn(x)
        a_emb = self.a_linear(a)
        return self.T(torch.cat([z_x, a_emb], dim=-1))

    def forward(self, x, x_g):
        z_x = self.embed_fn(x)
        g = self.embed_fn(x_g).unsqueeze(1)
        a_embs = self.a_linear(torch.arange(self.act_dim).to(x.device)) \
            .repeat(z_x.size(0), 1, 1)
        z_xs = z_x.unsqueeze(1).repeat(1, self.act_dim, 1)
        z_x_nexts = self.T(torch.cat([z_xs, a_embs], dim=-1))
        return - torch.norm(z_xs - z_x_nexts, p=self.embed_p, dim=-1) - torch.norm(
            z_x_nexts - g, p=self.embed_p, dim=-1)


from plan2vec.models.resnet import ResNet18, ResNet18Coord


class ResNet18L2Q(Q):
    def __init__(self, input_dim, action_dim, latent_dim, p=2):
        super().__init__()
        self.p = p
        self.trunk = ResNet18(input_dim, latent_dim * action_dim)
        self.embed = Λ(
            lambda x: self.trunk(x.reshape(-1, *x.shape[-3:])).reshape(*x.shape[:-3], latent_dim, action_dim))
        self.head = Λ(lambda a, b: (a - b).norm(p, dim=-2))

    def forward(self, x, x_prime):
        # note: for some reason, if x_prime has batch size of 1, it works poorly with DataParallel.
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        _ = self.embed(x), self.embed(x_prime)
        return self.head(*_)


class ResNet18CoordL2Q(Q):
    def __init__(self, input_dim, action_dim, latent_dim, p=2):
        super().__init__()
        self.p = p
        self.trunk = ResNet18Coord(input_dim, latent_dim * action_dim)
        self.embed = Λ(
            lambda x: self.trunk(x.reshape(-1, *x.shape[-3:])).reshape(*x.shape[:-3], latent_dim, action_dim))
        self.head = Λ(lambda a, b: (a - b).norm(p, dim=-2))

    def forward(self, x, x_prime):
        # note: for some reason, if x_prime has batch size of 1, it works poorly with DataParallel.
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        _ = self.embed(x), self.embed(x_prime)
        return self.head(*_)


class ResNet18CoordQ(Q):
    def __init__(self, input_dim, action_dim, latent_dim, p=2):
        super().__init__()
        self.p = p
        self.embed = ResNet18Coord(input_dim, latent_dim)
        self.head = nn.Sequential(
            nn.Linear(latent_dim * 2, action_dim)
        )

    def forward(self, x, x_prime):
        # note: for some reason, if x_prime has batch size of 1, it works poorly with DataParallel.
        x, x_prime = torch.broadcast_tensors(x, x_prime)
        z, z_ = self.embed(x), self.embed(x_prime)
        return self.head(torch.cat([z, z - z_], dim=-1))


if __name__ == '__main__':
    import torch

    # test norm

    a = torch.ones(11, 3, 4)
    b = torch.randn_like(a)
    c = torch.norm(a - b, p=2, dim=1)
    print(c.shape)

if __name__ == '__main__':
    import torch

    x = torch.ones(11, 3, 1, 64, 64)
    g = torch.randn_like(x)
    resnet = ResNet18L2Q(1, 8, 2)

    qs = resnet(x, g)
    print(qs.shape)
