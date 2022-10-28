"""
New Plan2vec Implementation, uses graph-planning algorithms to navigate the graph.

Test if the value function blows up or not.
"""
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import gym
from params_proto import proto_partial
from params_proto.neo_proto import ParamsProto, Proto
import torch
from torch import nn

from plan2vec.mdp.replay_buffer import EpisodicBuffer, ReplayBuffer
from plan2vec.mdp.td_lambda import mc_target
from torch_utils import batchify


class Args(ParamsProto):
    # env_id = "CMazeDiscreteImgIdLess-v0"
    seed = 10
    env_id = "GoalMassDiscreteImgIdLess-v0"
    obs_keys = "x", "img"
    input_dim = 1
    latent_dim = 3
    act_dim = 8  # used by the advantage function

    # sample parameters
    n_envs = 10
    num_rollouts = 400
    limit = 2  # limit for the timestamp.

    # graph parameters
    neighbor_r = 0.036
    neighbor_r_min = None

    # search params
    h_scale = 3
    search_alg = "a_star"

    num_epochs = 1000
    batch_size = 10  # sampling batch_size

    # learning parameters
    optim_epochs = 10
    optim_batch_size = 16
    lr = 1e-4
    lam = 0.8
    gamma = 0.99
    global_metric = "ResNet18CoordL2"
    target_update = False

    adv_k = 1  # The goal relabel interval for the advantage learning
    adv_after = 80
    adv_lr = 1e-5

    device = None
    criteria = Proto("nn.MSELoss(reduction='mean')", help="evaluated to a criteria object")

    visualization_interval = 10
    eval_interval = 10
    checkpoint_after = 80
    checkpoint_interval = 20


@proto_partial(Args)
def sample_trajs(seed, env_id, num_rollouts, obs_keys, limit):
    from ge_world import IS_PATCHED
    assert IS_PATCHED, "required for these envs."

    np.random.seed(seed)
    env = gym.make(env_id)
    env.reset()

    trajs = []
    for i in range(num_rollouts):
        obs, done = env.reset(), False
        path = defaultdict(list, {k: [obs[k]] for k in obs_keys})
        while not done:
            action = np.random.randint(low=0, high=7)
            obs, reward, done, info = env.step(action)
            for k in obs_keys:
                path[k].append(obs[k])
            # info: add action to data set.
            path['a'].append(action)

            if limit and len(path[k]) >= limit:
                break
        trajs.append({k: np.stack(v, axis=0) for k, v in path.items()})

    from ml_logger import logger
    logger.print(f'seed {seed} has finished sampling.', color="green")
    return np.array(trajs)


# if __name__ == '__main__':
#     trajs = sample_trajs(seed=10, env_id="CMazeDiscreteIdLess-v0", obs_keys=("x",))
#     print(len(trajs))


def eval_policy(env, policy, obs_key, goal_key, num_rollouts, limit):
    from tqdm import trange
    from ml_logger import logger

    for _ in trange(num_rollouts, desc="evaluate"):
        obs, done = env.reset(), False
        i, R = 0, 0
        while not done:
            action = policy(obs[obs_key], obs[goal_key])
            obs, reward, done, info = env.step(action)
            R += reward
            if info['success'] or limit and i >= limit:
                break
            i += 1
        logger.store_metrics(success=info['success'], R=R, ep_len=i)

    logger.print(f'has finished evaluation.', color="green")


# if __name__ == '__main__':
#     from ge_world import IS_PATCHED
#     from ml_logger import logger
#
#     assert IS_PATCHED, "required for these envs."
#     import numpy as np
#
#     env = gym.make(Args.env_id)
#     action_map = {tuple(np.sign(a).tolist()): ind for ind, a in enumerate(env.a_dict)}
#     policy = lambda x, g: action_map[tuple(np.sign(g - x).tolist())]
#
#     eval_policy(env, policy, obs_key="x", goal_key="goal", num_rollouts=10, limit=20)
#     logger.peek_stored_metrics()
#     logger.log_metrics_summary()
#
#     pass


def l2_metric(a, b):
    return np.linalg.norm(a - b, ord=2)


def plot_trajectory_2d(path, color='black', **kwargs):
    for (x, y), (x_, y_) in zip(path[:-1], path[1:]):
        dx = (x_ - x)
        dy = (y_ - y)
        d = np.linalg.norm([dx, dy], ord=2)
        plt.arrow(x, y, dx * 0.8, dy * 0.8, **kwargs, head_width=d * 0.3, head_length=d * 0.3,
                  length_includes_head=True, head_starts_at_zero=True, fc=color, ec=color)


def set_fig():
    plt.xlim(-24, 24)
    plt.ylim(-24, 24)


class MazeGraph(nx.Graph):
    all_images = None

    def __init__(self, trajs, obs_keys, r, r_min=None, d=l2_metric):
        super().__init__()

        if 'img' in obs_keys:
            self.all_images = np.concatenate([p['img'] for p in trajs])

        self.all_positions = np.concatenate([p['x'] for p in trajs])

        self.transitions = dict()
        i = 0
        for p in trajs:
            l = len(p['x'])
            for n in range(l):
                if n < l - 1:
                    self.transitions[(i, i + 1)] = p['a'][n]
                i += 1

        for i, xy in enumerate(self.all_positions):
            self.add_node(i, pos=xy)

        # Here we iterate through the nodes to add the edges
        from tqdm import tqdm
        for i, a in tqdm(self.nodes.items(), desc="edges"):
            for j, b in self.nodes.items():
                l = d(a['pos'], b['pos'])
                if l < r and (r_min is None or l > r_min):
                    self.add_edge(i, j, weight=l)

        self.queries = defaultdict(lambda: 0)

    def pop_cost(self):
        cost = len(self.queries.keys())
        self.queries.clear()
        return cost

    def neighbors(self, n, log=True):
        ns = list(super().neighbors(n))
        for n in ns if log else []:
            self.queries[n] += 1
        return ns

    def sample(self, batch_n=1):
        return np.random.randint(0, len(self.nodes), size=batch_n)

    def add_trajs(self):
        raise NotImplementedError('allow dynamic add of new trajs.')
        return self

    def show(self, filename=None, show=True):
        print('showing the graph')
        plt.figure(figsize=(2, 2), dpi=120)
        nx.draw(self, [n['pos'] for n in self.nodes.values()],
                node_size=0, node_color="gray", alpha=0.7, edge_color="gray")
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        if filename:
            from ml_logger import logger
            logger.savefig(filename)
        if show:
            plt.show()

    def closest(self, x, y, local_metric):
        ds = np.array([local_metric(n['pos'], [x, y]) for i, n in self.nodes.items()])
        return np.argmin(ds)

    def ind2pos(self, inds, scale=1):
        return [self.nodes[n]['pos'] * scale for n in inds]

    def plot_traj(self, path, show=True, title=None):
        if show:
            plt.figure(figsize=(2, 2), dpi=120)
        if title:
            plt.title(title)
        plot_trajectory_2d(self.ind2pos(path, 100))
        plt.scatter(*zip(*self.ind2pos(self.queries.keys(), 100)), color="gray", s=3, alpha=0.6)
        plt.gca().set_aspect('equal')
        set_fig()
        plt.tight_layout()
        if show:
            plt.show()


class Advantage(nn.Module):

    def __init__(self, metric_fn, act_dim):
        super().__init__()
        # avoid attaching the params.
        self.embed = lambda x: metric_fn.embed(x)
        self.head = nn.Sequential(
            nn.Linear(metric_fn.latent_dim * 2, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, act_dim),
        )

    def forward(self, img, goal):
        z, z_goal = self.embed(img), self.embed(goal)
        _ = torch.cat([z, z - z_goal], -1).detach()
        return self.head(_)

    def hard(self, img, goal):
        logits = self(img, goal)
        return torch.argmax(logits, -1)

    def soft(self, img, goal):
        logits = self(img, goal)
        return torch.distributions.Categorical(logits=logits).sample()


def compute_adv_loss(adv_fn, v_fn, all_images, state, next, goal, a, **_):
    from ml_logger import logger
    s = torch.tensor(all_images[state]).to(Args.device).float()
    s_ = torch.tensor(all_images[next]).to(Args.device).float()
    g = torch.tensor(all_images[goal]).to(Args.device).float()
    a = torch.tensor(a).to(Args.device).long()
    # of course this is not going to be very stable, but let's try it.
    all_values = adv_fn(s, g)
    adv = all_values.gather(-1, a[:, None]).squeeze(-1)
    adv_target = v_fn(s_, g).detach() - v_fn(s, g).detach()
    adv_values_mean = all_values.mean(dim=-1)

    # bp = []
    # for ns, g_ in zip(neighbors, g):
    #     logger.store_metrics(num_neighbors=len(ns))
    #     ns = torch.tensor(all_images[ns]).to(Args.device).float()
    #     bp.append(v_fn(ns, g_[None, ...]).mean().unsqueeze(0))
    # bp = torch.cat(bp)

    with torch.no_grad():
        logger.store_metrics(adv_target=adv_target.detach().mean().cpu().item(),
                             # neighbor_mean=bp.cpu().mean().item(),
                             adv_values=all_values.mean().cpu().item())

    return adv, adv_target, adv_values_mean, torch.zeros_like(adv_values_mean)


def train(deps=None):
    from graph_search import methods
    from ml_logger import logger
    from torch import optim, nn
    from torch_utils import torchify
    from tqdm import trange
    from plan2vec.models.resnet import ResNet18CoordL2
    from plan2vec.plotting.maze_world.embedding_image_maze import cache_images, \
        visualize_embedding_2d_image, visualize_embedding_3d_image, visualize_value_map

    Args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Args._update(deps)
    logger.log_params(Args=vars(Args))

    trajs = sample_trajs(seed=Args.seed, env_id=Args.env_id, num_rollouts=Args.num_rollouts, obs_keys=Args.obs_keys)
    logger.print('finished sampling', color="green")

    graph = MazeGraph(trajs, obs_keys=Args.obs_keys, r=Args.neighbor_r, r_min=Args.neighbor_r_min)
    graph.show("figures/sample_graph.png")

    # todo: add the Q-learning stuff.
    search = methods[Args.search_alg]

    Φ = eval(Args.global_metric)(
        Args.input_dim, latent_dim=Args.latent_dim).to(Args.device)
    # target_Φ = eval(Args.global_metric)(
    #     Args.input_dim, latent_dim=Args.latent_dim).to(Args.device)
    # target_Φ.load_state_dict(Φ.state_dict())
    # target_Φ.eval()

    adv_fn = Advantage(Φ, act_dim=Args.act_dim).to(Args.device)
    adv_fn_eval = batchify(torchify(adv_fn.hard, device=Args.device, dtype=torch.float32))

    d = torchify(Φ, dtype=torch.float32, device=Args.device, input_only=True)
    value_fn = lambda *_: - d(*_).squeeze()
    target_value_fn = lambda *_: - d(*_).squeeze().cpu().numpy()

    # gets path, ds
    # pick first pair (s, s', a)
    # A(s, a, g) = V(s', g) - V(s, g)
    # advantage =

    heuristic = lambda a, b: d(graph.all_images[a], graph.all_images[b]).squeeze().cpu().numpy() * Args.h_scale
    criteria = eval(Args.criteria)
    optimizer = optim.Adam(Φ.parameters(), lr=Args.lr)
    adv_optimizer = optim.Adam(adv_fn.parameters(), lr=Args.adv_lr)

    # graph.plot_traj(path, title="CMaze")

    logger.print('start learning', color="yellow")
    # Implement multi-step value function
    # todo: test the target value vs ground-truth
    buffer = EpisodicBuffer(20000)
    adv_buffer = ReplayBuffer(10000)
    for epoch in range(Args.num_epochs):
        # sampling from the graph
        for i in trange(Args.batch_size, desc="sample"):
            while True:
                try:
                    start, goal = graph.sample(2)
                    with torch.no_grad():
                        path, ds = search(graph, start, goal, heuristic)
                    # make freaking flow map.
                    logger.store_metrics(cost=graph.pop_cost(), path_length=np.sum(ds), plan_steps=len(ds))
                    buffer.extend(start=start, goal=goal, path=path, ds=np.array(ds))
                    break
                except Exception as e:
                    print(f'planning failed {start}, {goal}')
                    pass

            for (p, p_), act in graph.transitions.items():
                for g in graph.sample(5):
                    # pass in the neighbors to generate the back pressure for the other logits.
                    neighbors = graph.neighbors(p, log=False)  # we do not count these calls.
                    adv_buffer.add(state=p, a=act, next=p_, goal=g, neighbors=neighbors)

        for i in trange(Args.optim_epochs, desc="optim"):
            samples = buffer.sample(Args.optim_batch_size)

            states = graph.all_images[[traj['path'][0] for traj in samples]]
            goal_states = graph.all_images[[traj['goal'] for traj in samples]]
            values = value_fn(states, goal_states)
            target_values = torch.tensor([
                -mc_target(traj['ds']) for traj in samples],
                dtype=torch.float32, device=Args.device)

            # # single rollout math
            # rewards = - np.array(ds)
            # states = graph.all_images[path]
            # goal_state = graph.all_images[goal]
            #
            # # this is where the training happen
            # with torch.no_grad():
            #     target_value = torch.tensor(mc_target(rewards), dtype=torch.float32)
            #     # _ = td_target(states[1:], rewards, goal_state, target_value_fn, Args.lam, Args.gamma)
            #     # target_value = torch.tensor(_, dtype=torch.float32)

            loss = value_loss = criteria(values, target_values)

            action_samples = adv_buffer.sample(Args.optim_batch_size)
            if action_samples and epoch > (Args.adv_after or 0):
                adv_value, adv_target, adv_mean, neighbor_mean = \
                    compute_adv_loss(adv_fn, value_fn, graph.all_images, **action_samples)
                adv_loss = criteria(adv_value, adv_target)
                back_pressure_loss = criteria(adv_mean, neighbor_mean)
                loss += adv_loss + 10 * back_pressure_loss
                logger.store_metrics(adv_loss=adv_loss.cpu().item(),
                                     back_pressure_loss=back_pressure_loss.cpu().item(), )

            loss.backward()
            optimizer.step()

            if action_samples and epoch > (Args.adv_after or 0):
                adv_optimizer.step()

            logger.store_metrics(loss=loss.cpu().item(),
                                 value_loss=value_loss.cpu().item(),
                                 value=values.mean().cpu().item(), )
        logger.log_metrics_summary(key_values=dict(epoch=epoch))

        # if not Args.target_update:
        #     pass
        # elif Args.target_update < 1:  # <-- soft target update
        #     params = Φ.state_dict()
        #     with torch.no_grad():
        #         for name, param in target_Φ.state_dict().items():
        #             param.copy_((1 - Args.target_update) * params[name] + Args.target_update * param)
        # elif epoch % Args.target_update == 0:  # <-- hard target update
        #     target_Φ.load_state_dict(Φ.state_dict())

        if epoch % Args.visualization_interval == 0:
            cache_images(Args.env_id)
            eval(f"visualize_embedding_{Args.latent_dim}d_image")(  # png version for quick view
                Args.env_id, Φ.embed, f"figures/embedding/embed_{Args.latent_dim}d_{epoch:04d}.png")

        if not Args.eval_interval or epoch < (Args.adv_after or 0):
            pass
        elif epoch % Args.eval_interval == 0:
            if 'eval_env' not in locals():
                eval_env = gym.make(Args.env_id)
            eval_policy(eval_env, adv_fn_eval, 'img', 'goal_img', num_rollouts=50, limit=20)

        if not Args.checkpoint_interval or epoch < (Args.checkpoint_after or 0):
            pass
        elif epoch % Args.checkpoint_interval == 0:
            logger.save_module(adv_fn, f'models/{epoch:04d}/adv.pkl', chunk=20_000)
            logger.save_module(Φ, f'models/{epoch:04d}/Φ.pkl', chunk=10_000_000)
            logger.print(f"model is saved.", color="green")


if __name__ == '__main__':
    import jaynes
    from params_proto.neo_hyper import Sweep
    from plan2vec_experiments import instr, config_charts
    from ml_logger import logger

    jaynes.config()

    with Sweep(Args) as sweep:
        # Args.env_id = "CMazeDiscreteImgIdLess-v0"
        Args.env_id = "GoalMassDiscreteImgIdLess-v0"
        Args.latent_dim = 2
        # Args.num_rollouts = 400
        Args.num_rollouts = 400
        # Args.search_alg = ["dijkstra", "a_star", "heuristic"]
        Args.search_alg = "dijkstra"
        with sweep.product:
            Args.adv_lr = [1e-3, 1e-4, 3e-5, 1e-5, 3e-6]

    for deps in sweep:
        thunk = instr(train, deps, __prefix="advantage-zero-average",
                      __postfix=f"{Args.search_alg}-adv_lr-{Args.adv_lr}")
        jaynes.run(thunk, )
        config_charts("""
charts:
- yKey: value_loss/mean
  xKey: epoch
- yKey: adv_loss/mean
  xKey: epoch
- yKey: adv_target/mean
  xKey: epoch
- yKey: neighbor_mean/mean
  xKey: epoch
- yKey: adv_values/mean
  xKey: epoch
- yKey: value/mean
  xKey: epoch
- yKey: success/mean
  xKey: epoch
- yKey: cost/mean
  xKey: epoch
- type: image
  glob: "**/sample_graph.png"
- type: image
  glob: "figures/embedding/*.png"
keys:
- run.status
- Args.n_rollouts
            """)
        logger.log_text(__doc__, "README.md")

    jaynes.listen(600)
