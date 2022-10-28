"""
New Plan2vec Implementation, uses graph-planning algorithms to navigate the graph.

We multiply the ground-truth step size by 34, so that it falls inbetween [0.85 and 1.2].

We use this directly as the reward.

Freeze the learning rate after advantage training starts.
"""
from collections import defaultdict
from os.path import expanduser

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from params_proto import proto_partial
from params_proto.neo_proto import ParamsProto, Proto
from torch import nn

from plan2vec.mdp.replay_buffer import EpisodicBuffer, ReplayBuffer
from plan2vec.mdp.td_lambda import mc_target
from torch_utils import batchify, torchify, Eval, tslice, View
from tqdm import trange


class Args(ParamsProto):
    seed = 10

    # env_id = "GoalMassDiscreteImgIdLess-v0"
    sample_env_id = None
    env_id = "CMazeDiscreteImgIdLess-v0"

    obs_keys = "x", "img"
    input_dim = 1
    latent_dim = 10
    act_dim = 8  # used by the advantage function

    env_pos = None
    env_goal = None
    dump_plans = False

    # sample parameters
    num_evals = 10  # used only by planning evaluation
    num_eval_rollouts = 20
    num_rollouts = 200
    limit = 5  # limit for the timestamp.
    eval_limit = 20
    no_neighbor = True

    # graph parameters
    prune_r = 0.4
    neighbor_r = 1.
    neighbor_r_min = 0.5

    # this is the fastest
    local_metric = "LocalMetricCoordConv"
    load_local_metric = None

    # search params
    h_scale = 3
    search_alg = "dijkstra"

    num_epochs = 500
    batch_size = 10  # sampling batch_size

    # learning parameters
    optim_epochs = 10
    optim_batch_size = 16
    lr = 1e-5
    lam = 0.8
    gamma = 0.99
    global_metric = "ResNet18CoordL2"
    start_epoch = None
    load_global_metric = Proto(None, help="path for pre-trained global metric weights Φ")
    load_adv = Proto(None, help="The pre-trained advantage load path")
    # start_epoch = 801
    # load_global_metric = f"/geyang/plan2vec/2020/01-23/plan2vec/maze_plan2vec/advantage-centroid-loss/18.03/dijkstra-lr-1e-05-ctr-0.001/56.741100/models/{start_epoch - 1:04d}/Φ.pkl"
    # load_adv = f"/amyzhang/plan2vec/2020/01-27/neo_plan2vec/tweaking_maze_adv/maze_load_adv_tweak/bp-tweak-fix-sign/11.26/GoalMassDiscreteImgIdLess-v0-soft/27.887926/models/1800/adv.pkl"
    target_update = False

    centroid_scale = 0.001  # 0.001 and 0.003 works with lr = 1e-5
    centroid_criteria = "nn.MSELoss(reduction='mean')"

    adv_k = 1  # The goal relabel interval for the advantage learning
    adv_after = 800
    adv_lr = 1e-5
    adv_mean_scale = 0.1
    adv_mean_target = Proto("v_fn_eval(s, s_) * 2", help="this is a string that is evaluated in situ as the target.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criteria = Proto("nn.MSELoss(reduction='mean')", help="evaluated to a criteria object")

    eval_soft = True

    graph_z_update_interval = None
    visualization_interval = 10
    eval_interval = 10
    checkpoint_after = 400
    checkpoint_overwrite = True
    checkpoint_interval = None


def l2(a, b):
    import numpy as np
    return np.linalg.norm(a - b, ord=2)


# Note: what to do with edges that the agent is not able to reach?
#  1-unreachable.
#  Local metric Learns from buffer: x, x', d. d ∋ [0, 1, 2]. We set the threshold
#  to 1.1, so anything that is over is pruned from the graph.
#  ~
#  But this way is slow. We need the graph

@proto_partial(Args)
def sample_trajs(seed, env_id, num_rollouts, obs_keys, limit):
    from tqdm import trange
    from ge_world import IS_PATCHED
    assert IS_PATCHED, "required for these envs."

    np.random.seed(seed)
    env = gym.make(env_id)
    env.seed(seed)

    trajs = []
    for i in trange(num_rollouts):
        obs, done = env.reset(), False
        old_obs = obs
        path = defaultdict(list, {k: [obs[k]] for k in obs_keys})
        while not done:
            action = np.random.randint(low=0, high=7)
            obs, reward, done, info = env.step(action)
            for k in obs_keys:
                path[k].append(obs[k])
            path['r'].append(- l2(obs['x'], old_obs['x']) * 34)  # info: 34 makes the reward range [0.85, 1.2].
            path['a'].append(action)  # info: add action to data set.
            old_obs = obs

            if limit and len(path[k]) >= limit:
                break
        trajs.append({k: np.stack(v, axis=0) for k, v in path.items()})

    from ml_logger import logger
    logger.print(f'seed {seed} has finished sampling.', color="green")
    return np.array(trajs)


from plan2vec.plan2vec.maze_plan2vec import eval_policy


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


from more_itertools import chunked


def pairwise_fn(φ, xs, xs_slice, chunk=3, row_chunk=8000):
    from tqdm import tqdm

    return torch.cat([
        torch.cat([
            φ(*torch.broadcast_tensors(_xs.unsqueeze(0), _xs_slice.unsqueeze(1))).squeeze(-1)
            for _xs_slice in tqdm(tslice(xs_slice, chunk=chunk), desc="computing pair-wise distance")
        ]) for _xs in tslice(xs, chunk=row_chunk)], dim=-1)


class EdgeView:
    def __init__(self, pairwise, edge_mask):
        self.pairwise = pairwise
        self.edge_mask = edge_mask

    def __getitem__(self, item):
        i, j = item
        ok = self.edge_mask[i, j]
        return dict(weight=self.pairwise[i, j]) if ok else {}

    def __call__(self, r=None, r_min=None):
        mask = self.edge_mask.copy()
        if r is not None:
            mask &= self.pairwise < r
        if r_min is not None:
            mask &= self.pairwise > r_min

        return np.argwhere(mask)


# 1. I have a bunch of images
# 2. add image and embeddings to graph
# 3. edges are generated on the fly.
# What does it mean that the low-level policy is not able to "go through"?
# give: (s, g) -> a. env.step(a) -> s', d(s', g) > min_r. Remove this edge. rewards[s][g] -> 2.
class ImageMazeGraph:
    images = None
    positions = None
    index = None
    pairwise = None
    edge_mask = None

    def __init__(self, local_metric, r=None, r_min=None):
        self.local_metric = local_metric
        self.neighbor_r = r
        self.neighbor_r_min = r_min

        self.rewards = defaultdict(dict)
        self.queries = 0

    @torchify(dtype=torch.float32, device=lambda: Args.device)
    def d_eval(self, *_):
        with Eval(self.local_metric), torch.no_grad():
            return self.local_metric(*_).squeeze()

    def __len__(self):
        return 0 if self.images is None else len(self.images)

    # 1-part identity, 1-part nearest neighbor, 1-part far-away neighbors
    def paired_dataset(self, batch_size, distance=False):
        ss, ns, rs = [], [], []
        for s, d in self.rewards.items():
            for n, r in d.items():
                ss.append(s)
                ns.append(n)
                rs.append(r)
        ss, ns, rs = [np.array(a) for a in [ss, ns, rs]]
        inds = np.random.permutation(len(ss))
        shuffled = np.random.permutation(len(ss))

        for chunk, rand_chunk in zip(chunked(inds, batch_size),
                                     chunked(shuffled, batch_size)):
            yield ss[chunk], ns[chunk], ss[rand_chunk], - rs[chunk] if distance else rs[chunk]

    def sample(self, n=1):  # how abt without replacement?
        return np.random.randint(0, len(self.index), size=n)

    def add_node(self, img):
        pass

    def add_trajs(self, trajs, obs_key, reward_key, pos_key="x"):
        xs, l = [] if self.images is None else [self.images], len(self)

        if pos_key is not None:
            poses = [] if self.positions is None else [self.positions]

        for traj in trajs:
            xs.append(traj[obs_key])
            if pos_key is not None:
                poses.append(traj[pos_key])
            for ind, r in enumerate(traj[reward_key]):
                self.rewards[l + ind][l + ind + 1] = r
            l += len(xs[-1])

        self.images = np.concatenate(xs)
        self.index = np.arange(len(self.images), dtype=int)
        self.positions = np.concatenate(poses)

    def remove_node(self, n):
        ind = torch.arange(len(self)) != n
        self.images = self.images[ind]
        self.index = np.arange(len(self))
        self.zs = self.zs[ind]
        self.zs_2 = self.zs_2[ind]
        del self.rewards[n]

    @property
    def edges(self):
        return EdgeView(self.pairwise, self.edge_mask)

    def neighbors(self, n, log=True):
        if log:
            self.queries += 1

        # KISS implementation for the win!!
        row = self.pairwise[n]
        mask = self.edge_mask[n].copy()
        if self.neighbor_r:
            mask &= row < self.neighbor_r
        if self.neighbor_r_min:
            mask &= row > self.neighbor_r_min

        ds = row[mask]
        ord = np.argsort(ds)
        return self.index[mask][ord]

    def prune(self, n, m):
        # None: keep pairwise.
        # self.pairwise[n, m] = 2
        self.edge_mask[n, m] = False

    def query(self, image, r=None, r_min=None, log=False):
        """query nodes with an image"""
        # No need to cache, used to find identical images in memory.
        # you should use r as r_min otherwise
        ds = self.d_eval([image], self.images)
        mask = np.full(ds.shape, True)
        if r is not None:
            mask &= ds < r
        if r_min is not None:
            mask &= ds > r_min

        if log:
            self.queries += 1

        selected_ds = ds[mask]
        selected_inds = self.index[mask]
        order = np.argsort(selected_ds)
        return selected_inds[order], selected_ds[order]

    def closest(self, img, r=None, r_min=None, log=False):
        ns, ds = self.query(img, r, r_min, log)
        if len(ns):
            return ns[0], ds[0]
        return None, None

    def compute_pairwise(self, cache_seed=None):
        """

        :param cache_seed:
        :param no_cache: requires re-compute.
        :return:
        """
        with torch.no_grad():
            self.local_metric.eval()
            _ = torch.tensor(self.images, device=Args.device, dtype=torch.float32)
            pairwise_ds = pairwise_fn(self.local_metric, _, _, chunk=1).cpu().numpy()

            from ml_logger import ML_Logger  # , logger as l
            if cache_seed:
                l = ML_Logger(log_directory=expanduser("~/.plan2vec"), prefix="")
                filename = f"data/{cache_seed}/pairwise.pkl"
                l.print('pairwise shape: ', pairwise_ds.shape, color="green")
                l.log_data(pairwise_ds, filename, overwrite=True)
                l.print('finished logging pairwise data.')
                l.print(f'there are {(pairwise_ds < 1).sum()} pairs in the dataset.', flush=True)

        return pairwise_ds

    def miracle(self, cache_seed=None, no_cache=True):
        """I need a bit of miracle in my life."""
        from ml_logger import ML_Logger

        try:
            assert not no_cache
            l = ML_Logger(log_directory=expanduser("~/.plan2vec"), prefix="")
            filename = f"data/{cache_seed}/pairwise.pkl"
            self.pairwise, = l.load_pkl(filename)
            self.edge_mask = np.full(self.pairwise.shape, True)
            l.print('pairwise shape: ', self.pairwise.shape, color="green")
            return
        except:
            self.pairwise = self.compute_pairwise(cache_seed)
            self.edge_mask = np.full(self.pairwise.shape, True)

    def show(self, r=None, r_min=None, filename=None):
        """to show the graph"""
        from ml_logger import logger
        import matplotlib.pyplot as plt
        from tqdm import tqdm

        assert self.pairwise is not None, "you need to call miracle."

        edges = self.edges(r, r_min)

        plt.figure(figsize=(2.4, 2.8))
        for j, k in tqdm(edges[::50], desc="plotting"):
            plt.plot(*self.positions[[j, k]].T, color="red")

        logger.print('We thin the connections by 50x to speed up plotting.', color='green')

        plt.gca().set_aspect('equal')
        plt.axis('off')
        plt.tight_layout()
        if filename:
            logger.savefig(filename)
        plt.show()
        plt.close()


class Advantage(nn.Module):

    def __init__(self, input_dim, latent_dim, act_dim, ):
        super().__init__()
        # from plan2vec.models.resnet import ResNet18
        # self.embed = ResNet18(input_dim, latent_dim)
        self.act_dim = act_dim
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
            nn.Linear(100, act_dim)
        )

    def forward(self, img, goal):
        x, x_prime = torch.broadcast_tensors(img, goal)
        _ = torch.cat([x, x_prime], dim=-3)
        *b, C, H, W = _.shape
        return self.trunk(_.reshape(-1, C, H, W)).reshape(*b, self.act_dim)

    def hard(self, img, goal):
        logits = self(img, goal)
        return torch.argmax(F.softmax(logits, -1), -1)

    def soft(self, img, goal):
        logits = self(img, goal)
        return torch.distributions.Categorical(logits=logits * 8).sample()


def compute_adv_loss(adv_fn, v_fn_eval, all_images, state, next, goal, a, neighbors=None):
    # Note:
    #  if d(s, g) - d(s, s_) - d(s_, g) is the advantage for choosing s_. It is always negative
    #  but it becomes 0 if and only if s_ is on the line in-between s and g.
    #  v = -d, advantage is always negative, only zero when
    #  things are perfectly aligned. So it should be d(s, g) - [ d(s, s_) + d(s_, g)]
    #  which is below:
    from ml_logger import logger
    s, s_, g = all_images[state], all_images[next], all_images[goal]
    a = torch.tensor(a).to(Args.device).long()

    # These variables requires grad.
    all_values = adv_fn(s, g)
    adv_mean = all_values.mean(-1)
    adv_act = all_values.gather(-1, a[:, None]).squeeze(-1)  # should not subtract average

    with torch.no_grad():

        state_value = v_fn_eval(s, g)
        adv_target = v_fn_eval(s, s_) + v_fn_eval(s_, g) - state_value

        if Args.adv_mean_target:
            adv_mean_target = eval(Args.adv_mean_target)
        else:
            adv_mean_target = []  # note: Compute back-pressure.
            for i, (si, ns, gi) in enumerate(zip(s, neighbors, g)):
                logger.store_metrics(num_neighbors=len(ns))
                bow = v_fn_eval(si[None, ...], all_images[ns]) + v_fn_eval(all_images[ns], gi[None, ...])
                adv_mean_target.append(bow.mean(0, keepdim=True))

            adv_mean_target = torch.cat(adv_mean_target) - state_value

        logger.store_metrics(
            adv_act=adv_act.mean().cpu().item(),
            adv_target=adv_target.mean().cpu().item(),
            adv_values=all_values.mean().cpu().item(),
            adv_back_pressure=adv_mean_target.mean().cpu().item(),
            # adv_entropy=entropy.cpu()
        )

    return adv_act, adv_target, adv_mean, adv_mean_target


class Unroll:
    def __init__(self, env, policy_fn, obs_key, goal_key):
        self.env = env
        self.policy_fn = policy_fn

    def run_episode(self, limit=None):
        from ml_logger import logger
        t, obs = 0, self.env.reset()
        while True:
            a = self.policy_fn(obs[self.obs_key], obs[self.goal_key])
            obs, reward, done, info = self.env.step(a)

            if info['success'] or (limit and t > limit):
                logger.store_metrics(success=info.get('success', 0))


class PlanningPolicy:
    last_frame = None
    last_next_frame = None
    last = None
    last_next = None

    # debug flag
    dump_plans = True
    count = 0

    def __init__(self, graph: ImageMazeGraph, search, policy_fn, prune_r=None):
        self.graph = graph
        self.search = search
        self.policy_fn = policy_fn
        self.frame_buffer = []
        self.prune_r = prune_r

    def __call__(self, img, goal):
        """
        to plot the path:

        plt.plot(*self.graph.positions[path].T, 'o-', color="#23aaff", alpha=0.6,
                 linestyle='--', markeredgecolor='white', linewidth=3)
        """
        graph = self.graph

        # prune the graph:
        if self.prune_r is not None:
            from termcolor import cprint

            # if self.last_frame is not None:
            #     mov = self.graph.d_eval([img], [self.last_frame])

            if self.last_next_frame is not None:
                𝝳s, r2g = self.graph.d_eval([self.last_frame, img],
                                             [img, self.last_next_frame])
                if 𝝳s < self.prune_r:
                    # if r2g > self.prune_r or 𝝳s < self.prune_r:
                    self.graph.prune(self.last, self.last_next)
                    self.last_next = None
                    from ml_logger import logger
                    cprint(f'I am stuck. pruning the edge: {𝝳s:02f}-{r2g:02f}', color="red")
                else:
                    print(f'Not stuck: {𝝳s:02f}-{r2g:02f}')

        n, distance = graph.closest(img, log=False)
        goal_ind, g_distance = graph.closest(goal, log=False)
        path, 𝝳s = self.search(graph, n, goal_ind)

        if self.dump_plans:  # very slow :)
            from ml_logger import logger
            i, j = np.nonzero(np.logical_not(graph.edge_mask))
            a, b = graph.positions[i], graph.positions[j]
            logger.log_data(dict(
                path=path, ds=𝝳s, removed_edges=[a, b]
            ), path=f"plan_dumps/{self.count:04d}_data.pkl")
            self.count += 1

            import matplotlib.pyplot as plt
            plt.plot(*graph.positions[path].T, "o-", markersize=11, mec='white',
                     color='gray', alpha=0.7, linewidth=3)
            plt.gca().set_aspect('equal')
            plt.ylim(-0.26, 0.26)
            plt.xlim(-0.26, 0.26)
            logger.savefig(key=f"plan_dumps/{self.count:04d}_plan.png")
            logger.savefig(key=f"plan_dumps/{self.count:04d}_plan.pdf")
            plt.show()
            plt.close()

        self.last = n
        self.last_frame = img
        if path:
            self.last_next = path[1]
            self.last_next_frame = graph.images[path[1]]
            return self.policy_fn(img, graph.images[path[1]])
        else:
            return np.random.randint(Args.act_dim)


def eval_planning_policy(deps=None):
    from graph_search import methods
    from ml_logger import logger
    from torch_utils import torchify
    from tqdm import trange

    Args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Args._update(deps)
    logger.log_params(Args=vars(Args))

    try:
        logger.upload_file(__file__)
    except:
        pass

    logger.print('start sampling')
    trajs = sample_trajs(seed=Args.seed, env_id=Args.sample_env_id or Args.env_id, num_rollouts=Args.num_rollouts,
                         obs_keys=Args.obs_keys, limit=Args.limit)
    logger.print('finished sampling', color="green")

    from plan2vec.models.convnets import LocalMetricConv, LocalMetricCoordConv, \
        LocalMetricConvLarge, LocalMetricConvXL, LocalMetricConvDeep, LocalMetricConvL2, \
        LocalMetricConvLargeL2

    from plan2vec.models.resnet import ResNet18Stacked

    local_metric = eval(Args.local_metric)(Args.input_dim, latent_dim=Args.latent_dim).to(Args.device)
    logger.log_text(str(local_metric), "models/f_local_metric.txt")
    if Args.load_local_metric:
        logger.load_module(local_metric, Args.load_local_metric)

    graph = ImageMazeGraph(local_metric, Args.neighbor_r, Args.neighbor_r_min)

    # graph.add_trajs(trajs, obs_key, goal_key, reward_key)
    graph.add_trajs(trajs, "img", "r", "x")
    graph.miracle(cache_seed=Args.seed)
    graph.show(r=1, r_min=0.5, filename=f"figures/connection_({0.5}-{1}).png")
    graph.show(r=0.5, filename=f"figures/connection_({0.5}).png")
    # exit()

    # todo: add the Q-learning stuff.
    search = methods[Args.search_alg]

    adv = Advantage(Args.input_dim, Args.latent_dim, Args.act_dim).to(Args.device)
    if Args.load_adv:
        logger.load_module(adv, Args.load_adv)

    @batchify
    @torchify(device=Args.device, dtype=torch.float32)
    def adv_fn_eval(*_):
        with Eval(adv), torch.no_grad():
            return adv.hard(*_)

    planning_pi = PlanningPolicy(graph, search, adv_fn_eval, prune_r=Args.prune_r)
    # vis mode
    planning_pi.dump_plans = Args.dump_plans

    env = gym.make(Args.env_id)
    for i in trange(Args.num_evals, desc="evaluating planning policy"):
        eval_policy(env, planning_pi, "img", "goal_img", Args.num_eval_rollouts, limit=Args.eval_limit,
                    pos=Args.env_pos, goal=Args.env_goal,
                    filename=f"debug/planning_results_{i:02d}.mp4")
        logger.log_metrics_summary(key_values=dict(eval_step=i))

    logger.print('Planning Evaluation is finished!', color="green")

    exit()


# class f_config:


def train_local_metric(deps):
    from ml_logger import logger
    from torch import optim
    from torch_utils import torchify
    from tqdm import trange
    from plan2vec.plotting.maze_world.embedding_image_maze import cache_images

    Args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Args._update(deps)
    logger.log_params(Args=vars(Args))

    try:
        logger.upload_file(__file__)
    except:
        pass

    logger.print('start sampling')
    trajs = sample_trajs(seed=Args.seed, env_id=Args.env_id or Args.env_id, num_rollouts=Args.num_rollouts,
                         obs_keys=Args.obs_keys, limit=Args.limit)
    logger.print('finished sampling', color="green")

    # The L2 version works significantly better for picking out [0 - 1] transition
    from plan2vec.models.convnets import LocalMetricConvL2, LocalMetricConv, LocalMetricConvLargeL2
    from plan2vec.models.resnet import ResNet18Stacked
    local_metric = eval(Args.local_metric)(Args.input_dim, latent_dim=Args.latent_dim).to(Args.device)
    logger.log_text(str(local_metric), "models/f_local_metric.txt")
    if Args.load_local_metric:
        logger.load_module(local_metric, Args.load_local_metric)

    @batchify
    @torchify(device=Args.device, dtype=torch.float32, input_only=True)
    def d_eval(*_):
        with Eval(local_metric), torch.no_grad():
            return local_metric(*_).squeeze().item()

    graph = ImageMazeGraph(local_metric, Args.neighbor_r, Args.neighbor_r_min)

    graph.add_trajs(trajs, "img", "r", "x")

    import torch.nn.functional as F
    l1 = F.smooth_l1_loss

    # rev_hing_loss = lambda x, y: torch.max(y - x, 0, dim=-1)[0]

    def rev_hinge_loss(x, y: float):
        d = x - torch.ones_like(d_shuffle, requires_grad=False) * y
        return - (d * (d < 0).float()).mean()

    optimizer = optim.Adam(local_metric.parameters(), lr=Args.lr)

    all_images = torch.tensor(graph.images).float().to(Args.device)
    for epoch in trange(Args.num_epochs + 1, desc="training local metric"):
        for x, x_, x_perm, d in graph.paired_dataset(Args.batch_size, distance=True):
            d = torch.tensor(d).float().to(Args.device)
            d_bar = local_metric(all_images[x], all_images[x_]).squeeze()
            d_null = local_metric(all_images[x], all_images[x]).squeeze()
            d_shuffle = local_metric(all_images[x_perm], all_images[x_]).squeeze()

            loss = l1(d_bar, d) + l1(d_null, torch.zeros_like(d_null)) \
                   + rev_hinge_loss(d_shuffle, 2)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                logger.store_metrics(
                    loss=loss.mean().item(),
                    d=d.mean().item(),
                    d_bar=d_bar.mean().item(),
                    d_null=d_null.mean().item(),
                    d_shuffle=d_shuffle.mean().item())

        logger.log_metrics_summary(key_values=dict(epoch=epoch))

        if Args.graph_z_update_interval and epoch % Args.graph_z_update_interval == 0:
            graph.update_z()
            # graph.show()
        if Args.checkpoint_interval and epoch % Args.checkpoint_interval == 0:
            logger.save_module(local_metric, f"models/{epoch:04d}/f_lm.pkl", chunk=200_000_000)
            if Args.checkpoint_overwrite:
                logger.remove(f"models/{epoch - Args.checkpoint_interval:04d}/f_lm.pkl")


def train_adv():
    exit()

    @torchify(dtype=torch.float32, device=Args.device, input_only=True)
    def embed_fn_eval(*_):
        with Eval(Φ), torch.no_grad():
            return Φ.embed(*_)

    adv = Advantage(Args.input_dim, Args.latent_dim, Args.act_dim).to(Args.device)
    adv_fn = torchify(adv, device=Args.device, dtype=torch.float32, input_only=True)
    if Args.load_adv:
        logger.load_module(adv, Args.load_adv)

    @batchify
    @torchify(device=Args.device, dtype=torch.float32)
    def adv_fn_eval(*_):
        with Eval(adv), torch.no_grad():
            return adv.hard(*_)

    @torchify(device=Args.device, dtype=torch.float32, input_only=True)
    def value_fn(*_):  # Used for training Φ
        return - Φ(*_).squeeze()

    @torchify(device=Args.device, dtype=torch.float32, input_only=True)
    def embed_fn(*_):  # Used for training Φ
        return Φ.embed(*_)

    @torchify(device=Args.device, dtype=torch.float32, input_only=True)
    def value_fn_eval(*_):  # return torch.tensor for further processing
        with Eval(Φ), torch.no_grad():
            return - Φ(*_).squeeze()

    @torchify(device=Args.device, dtype=torch.float32)
    def heuristic(a, b):
        with Eval(Φ), torch.no_grad():
            _ = Φ(graph.images[a], graph.images[b])
            return _.squeeze() * Args.h_scale

    criteria = eval(Args.criteria)
    centroid_criteria = eval(Args.centroid_criteria)
    optimizer = optim.Adam(Φ.parameters(), lr=Args.lr)
    adv_optimizer = optim.RMSprop(adv_fn.parameters(), lr=Args.adv_lr)

    # graph.plot_traj(path, title="CMaze")

    logger.print('start learning', color="yellow")
    # Implement multi-step value function
    # todo: test the target value vs ground-truth
    buffer = EpisodicBuffer(20000)
    adv_buffer = ReplayBuffer(100000)
    for epoch in range(Args.start_epoch or 0, Args.num_epochs):
        # sampling from the graph
        for i in trange(Args.batch_size, desc="sample"):
            while True and (not Args.adv_after or epoch <= Args.adv_after):
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

            if Args.adv_after and epoch > Args.adv_after:
                for (p, p_), act in graph.transitions.items():
                    for g in graph.sample(5):
                        # pass in the neighbors to generate the back pressure for the other logits.

                        neighbors = None if Args.no_neighbor \
                            else graph.neighbors(p, log=False)  # we do not count these calls.
                        adv_buffer.add(state=p, a=act, next=p_, goal=g, neighbors=neighbors)

        for i in trange(Args.optim_epochs, desc="optim"):

            if epoch <= (Args.adv_after or 0):
                samples = buffer.sample(Args.optim_batch_size)

                states = graph.images[[traj['path'][0] for traj in samples]]
                goal_states = graph.images[[traj['goal'] for traj in samples]]
                values = value_fn(states, goal_states)
                target_values = torch.tensor([
                    -mc_target(traj['ds']) for traj in samples],
                    dtype=torch.float32, device=Args.device)

                loss = value_loss = criteria(values, target_values)

                random_obs = graph.images[graph.sample(20)]
                centroid = embed_fn(random_obs).mean(dim=0)
                centroid_loss = centroid_criteria(centroid, torch.zeros_like(centroid))

                loss += centroid_loss * Args.centroid_scale
                loss.backward()
                optimizer.step()
                logger.store_metrics(loss=loss.cpu().item(),
                                     value_loss=value_loss.cpu().item(),
                                     centroid_loss=centroid_loss.cpu().item(),
                                     value=values.mean().cpu().item(), )

            action_samples = adv_buffer.sample(Args.optim_batch_size)
            if action_samples and epoch > (Args.adv_after or 0):
                adv_value, adv_target, adv_mean, adv_mean_target = \
                    compute_adv_loss(adv_fn, value_fn_eval, graph.images, **action_samples)
                adv_loss = criteria(adv_value, adv_target)
                adv_mean_loss = criteria(adv_mean, adv_mean_target)

                logger.store_metrics(adv_loss=adv_loss.detach().cpu().numpy(),
                                     adv_mean_loss=adv_mean_loss.detach().cpu().numpy())

                (adv_loss + adv_mean_loss * Args.adv_mean_scale).backward()
                adv_optimizer.step()

        logger.log_metrics_summary(key_values=dict(epoch=epoch), default_stats="min_max")

        if epoch % Args.visualization_interval == 0 and Args.latent_dim <= 3:
            cache_images(Args.env_id)
            eval(f"visualize_embedding_{Args.latent_dim}d_image")(  # png version for quick view
                Args.env_id, embed_fn_eval, f"figures/embedding/embed_{Args.latent_dim}d_{epoch:04d}.png")

        if not Args.eval_interval or epoch < (Args.adv_after or 0):
            pass
        elif epoch % Args.eval_interval == 0:
            if 'eval_env' not in locals():
                eval_env = gym.make(Args.env_id)
            eval_policy(eval_env, adv_fn_eval, 'img', 'goal_img', num_rollouts=50, limit=20)

        if not Args.checkpoint_interval or epoch < (Args.checkpoint_after or 0):
            pass
        elif epoch % Args.checkpoint_interval == 0:
            if epoch <= Args.adv_after:
                logger.remove(f'models/{epoch - Args.checkpoint_interval:04d}/Φ.pkl')
                logger.save_module(Φ, f'models/{epoch:04d}/Φ.pkl', chunk=10_000_000)
            else:
                logger.remove(f'models/{epoch - Args.checkpoint_interval:04d}/adv.pkl')
                logger.save_module(adv_fn, f'models/{epoch:04d}/adv.pkl', chunk=10_000_000)
            logger.print(f"model is saved.", color="green")


def eval_advantage(env_id, weight_path):
    """this function is used to evaluate the advantage function as a stand-alone policy"""
    from ml_logger import logger
    from ge_world import IS_PATCHED

    import pandas as pd
    from plan2vec_experiments.analysis_icml_2020 import plot_bernoulli

    adv = Advantage(Args.input_dim, Args.latent_dim, Args.act_dim).to(Args.device)
    logger.load_module(adv, weight_path)

    @batchify
    @torchify(device=Args.device, dtype=torch.float32)
    def adv_fn_eval(*_):
        with Eval(adv), torch.no_grad():
            return adv.hard(*_)

    eval_env = gym.make(env_id)

    eval_policy(eval_env, adv_fn_eval, 'img', 'goal_img', num_rollouts=50, limit=20,
                filename="videos/game_play.mp4")

    # xKey, yKey = 'd2goal', 'success'
    # d = logger.summary_cache.data
    # df = pd.DataFrame({k: d[k] for k in [xKey, yKey]}).sort_values(xKey)
    # plot_bernoulli(df, xKey, yKey, figsize=(3.4, 2.2), filename="figures/success_vs_l2_d2g.png")


class IKDataset:
    inds = []
    next_inds = []

    def __init__(self, trajs, batch_size, obs_key='img', act_key='a'):
        self.batch_size = batch_size

        self.images = np.concatenate([p[obs_key] for p in trajs])
        self.actions = np.concatenate([p[act_key] for p in trajs])
        curr = 0
        inds, next_inds = [], []
        for p in trajs:
            l = len(p[obs_key])
            inds.append(range(curr, curr + l - 1))
            next_inds.append(range(curr + 1, curr + l))
            curr += l

        self.inds = np.concatenate(inds)
        self.next_inds = np.concatenate(next_inds)
        self.act_inds = np.arange(len(self), dtype=int)

    def __len__(self):
        return len(self.inds)

    def __iter__(self):
        from more_itertools import chunked
        ord = np.random.permutation(len(self.act_inds))

        for chunk in chunked(ord, self.batch_size):
            yield self.images[self.inds[chunk]], \
                  self.images[self.next_inds[chunk]], \
                  self.actions[self.act_inds[chunk]],


def train_advantage(env_id, deps=None, ):
    """"""
    from ml_logger import logger

    import pandas as pd
    from plan2vec_experiments.analysis_icml_2020 import plot_bernoulli

    Args._update(deps)
    Args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log_params(Args=vars(Args))

    adv = Advantage(Args.input_dim, Args.latent_dim, Args.act_dim).to(Args.device)
    if Args.load_adv:
        logger.load_module(adv, Args.load_adv)

    @torchify(device=Args.device, dtype=torch.float32)
    def adv_fn_eval(*_):
        with Eval(adv), torch.no_grad():
            return adv.hard(*_)

    adv_train = torchify(adv, dtype=torch.float32, input_only=True)

    trajs = sample_trajs(seed=Args.seed, env_id=env_id, num_rollouts=Args.num_rollouts, obs_keys=Args.obs_keys,
                         limit=Args.limit)
    dataset = IKDataset(trajs, Args.batch_size)

    test_trajs = sample_trajs(seed=100, env_id=env_id, num_rollouts=40,
                              obs_keys=Args.obs_keys, limit=Args.limit)
    test_dataset = IKDataset(test_trajs, Args.batch_size)

    optimizer = torch.optim.Adam(adv.parameters(), Args.adv_lr)
    # criteria = torch.nn.BCEWithLogitsLoss()
    criteria = torch.nn.CrossEntropyLoss()

    def evaluate_accuracy():
        for x, next, action in test_dataset:
            a_bar = adv_fn_eval(x, next)
            logger.store_metrics(accuracy=(a_bar == action).mean())

    for epoch in trange(Args.num_epochs + 1):
        for x, next, action in dataset:
            a_bar = adv_train(x, next)
            _action = torch.tensor(action).to(Args.device)
            loss = criteria(a_bar, _action)

            with torch.no_grad():
                life_is_hard = torch.argmax(a_bar, dim=-1).cpu().numpy()
                logger.store_metrics({"train/accuracy": (life_is_hard == action).mean()},
                                     loss=loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        evaluate_accuracy()
        logger.log_metrics_summary(key_values=dict(
            epoch=epoch, dt_epoch=logger.split()
        ))

        if Args.checkpoint_interval and epoch % Args.checkpoint_interval == 0:
            logger.save_module(adv, f"models/{epoch:04d}/adv.pkl", chunk=200_000_000)
            if Args.checkpoint_overwrite:
                logger.remove(f"models/{epoch - Args.checkpoint_interval:04d}/adv.pkl")

    print('training is complete')

    eval_env = gym.make(env_id)

    eval_policy(eval_env, adv_fn_eval, 'img', 'goal_img', num_rollouts=50, limit=20,
                filename="videos/game_play.mp4")

    # xKey, yKey = 'd2goal', 'success'
    # d = logger.summary_cache.data
    # df = pd.DataFrame({k: d[k] for k in [xKey, yKey]}).sort_values(xKey)
    # plot_bernoulli(df, xKey, yKey, figsize=(3.4, 2.2), filename="figures/success_vs_l2_d2g.png")


if __name__ == '__main__':
    import jaynes
    from params_proto.neo_hyper import Sweep
    from plan2vec_experiments import instr, config_charts
    from ml_logger import logger, cprint

    jaynes.config()

    with Sweep(Args) as sweep:
        Args.env_id = "CMazeDiscreteImgIdLess-v0"
        # Args.env_id = "GoalMassDiscreteImgIdLess-v0"
        Args.latent_dim = 2
        Args.num_rollouts = 400
        Args.search_alg = "dijkstra"
        Args.seed = 20
        with sweep.product:
            Args.adv_lr = [1e-5, 3e-6, 1e-6]

    for deps in sweep:
        thunk = instr(train, deps, __prefix="debug", )
        jaynes.run(thunk, )
        config_charts("""
            charts:
            - yKey: loss/mean
              xKey: epoch
            keys:
            - run.status
            - Args.n_rollouts
            """)
        logger.log_text(__doc__, "README.md")

    jaynes.listen(600)
