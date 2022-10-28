import torch
from torch import optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from params_proto import cli_parse, Proto
from termcolor import cprint
from tqdm import trange

from plan2vec.mdp.replay_buffer import ReplayBuffer
from plan2vec.plotting.visualize_traj_2d import visualize_prediction, local_metric_vs_ground_truth
from torch_utils import RMSLoss


class TripletLossDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


@cli_parse
class Args:
    env_id = 'GoalMassDiscreteIdLess-v0'
    seed = 0
    load_weights = Proto(None, help="path for trained weights", dtype=str)
    save_weights = Proto("models/local_metric.pkl", help="location to save the weight")

    num_envs = 20
    n_rollouts = 1000
    timesteps = 10

    num_epochs = 40
    batch_size = 32
    # lr = LinearAnneal(1e-3, min=1e-3, n=num_epochs + 1)
    lr = 1e-4
    k_fold = Proto(10, dtype=int, help="The k-fold for validation")

    vis_interval = 10


def train(memory: ReplayBuffer, metric_fn, eval_only=False):
    """Build dataset and train local metric for passive setting."""
    from ml_logger import logger

    batch = memory.sample(len(memory), 0)
    pos_labels = np.ones(len(memory))
    neg_labels = np.zeros(len(memory))

    # todo: consider supervising with actual distance.
    x = np.concatenate([batch['s'], batch['s']])
    x_prime = np.concatenate([batch['s_'], np.random.permutation(batch['s_'])])
    all_x = np.array(list(zip(x, x_prime)))
    all_y = np.concatenate([pos_labels, neg_labels])

    visualize_prediction(x, x_prime, all_y, k=20, key="figures/sample.png", title="Training Set")

    shuffle = np.random.rand(len(all_x)).argsort()
    valid_index = len(all_x) // Args.k_fold
    loader = DataLoader(TripletLossDataset(all_x[shuffle][:valid_index], all_y[shuffle][:valid_index]),
                        batch_size=Args.batch_size, shuffle=True, num_workers=4)
    eval_loader = DataLoader(TripletLossDataset(all_x[shuffle][valid_index:], all_y[shuffle][valid_index:]),
                             batch_size=valid_index, shuffle=True, num_workers=4)

    optimizer = optim.Adam(metric_fn.parameters(), lr=Args.lr)

    rms = RMSLoss()

    def evaluate():
        with torch.no_grad():
            for i, (_, y) in enumerate(eval_loader):
                s, s_prime = _[:, 0, :], _[:, 1, :]
                y_hat = metric_fn(s.float().to(Args.device), s_prime.float().to(Args.device)).squeeze(-1)
                correct = ((y_hat > 0.5) == y.byte()).sum().cpu().item()
                loss = rms(y_hat.view(-1), y.view(-1).float().to(Args.device))
                logger.store_metrics(accuracy=correct / len(y_hat), rms=loss.cpu().item())
                visualize_prediction(s.cpu().numpy(),
                                     s_prime.cpu().numpy(),
                                     y_hat.cpu().numpy(), k=20,
                                     title="Prediction",
                                     key=f"figures/pred_{epoch:04d}.png")
                local_metric_vs_ground_truth(s.cpu().numpy(),
                                             s_prime.cpu().numpy(),
                                             y_hat.cpu().numpy(),
                                             key=f"figures/score_vs_l1_{epoch:04d}.png")

    if eval_only:
        epoch = 0
        logger.load_module(metric_fn, Args.load_weights)
        evaluate()
        logger.log_metrics_summary(key_values=dict(eval_only=True), default_stats='mean')
        return

    for epoch in range(Args.num_epochs + 1):
        for i, (_, y) in enumerate(loader):
            s, s_prime = _[:, 0, :], _[:, 1, :]
            y_hat = metric_fn(s.float().to(Args.device), s_prime.float().to(Args.device))
            loss = F.smooth_l1_loss(y_hat.view(-1), y.view(-1).float().to(Args.device))
            logger.store_metrics(loss=loss.cpu().item())

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        evaluate()

        logger.log_metrics_summary(key_values=dict(epoch=epoch), default_stats='mean')

    if Args.save_weights:
        logger.save_module(metric_fn, "models/local_metric.pkl")


def main(freeze_after=True, **kwargs):
    from ml_logger import logger
    from plan2vec.mdp.wrappers.subproc_vec_env import SubprocVecEnv
    from plan2vec.mdp.helpers import make_env
    from plan2vec.mdp.sampler import path_gen_fn
    from plan2vec.models.mlp import LocalMetric

    Args.device = torch.device("cpu")
    Args.update(kwargs)

    logger.log_params(local_metric=vars(Args))

    np.random.seed(Args.seed)
    torch.manual_seed(Args.seed)

    envs = SubprocVecEnv([make_env(Args.env_id, Args.seed + i) for i in range(Args.num_envs)])
    logger.log_params(env=envs.spec._kwargs)

    memory = ReplayBuffer(Args.n_rollouts * Args.timesteps)

    random_pi = lambda ob, goal, *_: np.random.randint(0, 8, size=[len(ob)])
    random_path_gen = path_gen_fn(envs, random_pi, "x", "goal")
    next(random_path_gen)

    for i in trange(Args.n_rollouts // Args.num_envs):
        paths = random_path_gen.send(Args.timesteps)
        memory.extend(s=paths['obs']['x'].reshape(-1, 2), s_=paths['next']['x'].reshape(-1, 2), )
    envs.close()

    f_local_metric = LocalMetric(2, 10).to(Args.device)

    train(memory, f_local_metric, eval_only=Args.load_weights)

    if freeze_after:
        for p in f_local_metric.parameters():
            p.requires_grad = False

    return f_local_metric.eval()


if __name__ == "__main__":
    from ml_logger import logger
    from plan2vec_experiments import instr

    time = logger.now('%H.%M/%S.%f')
    _ = instr(main, __postfix="local-metric-experiments")

    if False:
        _(seed=5 * 100)

    if True:
        cprint('Training on cluster', 'green')
        import jaynes

        jaynes.config("devfair", display=logger.now("%f"))
        jaynes.run(_, seed=5 * 100)
        jaynes.listen()
