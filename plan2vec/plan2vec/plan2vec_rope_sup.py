from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from termcolor import cprint
from torch import optim, nn
import numpy as np
from params_proto import cli_parse, Proto
from heapq import heappush, heappop
from baselines.common.schedules import LinearSchedule

from torch_utils import torchify, tslice, sliced_helper
from plan2vec.mdp.models import LocalMetricConvLarge, LocalMetricConvDeep, GlobalMetricConvDeepKernel, GlobalMetricConvDeepL1Norm
from plan2vec.mdp.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

@cli_parse
class Args:
    seed = 0
    data_size = Proto(500, help="number of images to use")

    # Required data paths
    data_path = Proto("~/fair/new_rope_dataset/data/new_rope.npy", help="path to the data file")
    load_local_metric = Proto("models/local_metric.pkl", help="location to load the weight from")
    load_pairwise_ds = Proto(None, help="pre-computed distance between all samples, using the local metric")
    load_top_k = Proto(None, help="pre-computed top-k, using the local metric")
    load_neighbors = Proto(None, help="pre-computed neighbors, using the local metric")
    single_traj = Proto(None, help="train with only a single trajectory", dtype=int)
    double_dqn = Proto(False, dtype=bool, help="use Double DQN")
    alpha = Proto(0., dtype=float, help="alpha for prioritized replay")
    latent_dim = Proto(128, dtype=int, help="latent dimension size")

    term_r = Proto(1.3, help="the threshold value for a sample being considered a `neighbor`.")

    num_epochs = 5000
    batch_n = 200
    H = Proto(200, help="planning horizon for the policy")
    k = Proto(24, help="k neighbors")
    gamma = Proto(1, help="discount factor")

    r_scale = Proto(1, help="scaling factor for the reward")
    max_eps_greedy = Proto(0.1, help="epsilon greedy factor. Explore when rn < eps")
    min_eps_greedy = Proto(0.1)

    relabel_k = Proto(5, help="the k-interval for the relabeling")
    optim_steps = 5
    optim_batch_size = 256

    target_update = Proto(0.9, help="when less than 1, does soft update. When larger or equal to 1, does hard update.")

    lr = 0.01

    k_fold = Proto(10, help="The k-fold validation for evaluating the planning module.")
    evaluation_interval = 10
    visualization_interval = 1
    binary_reward = Proto(False, help="Uses a binary reward if true. Otherwise uses local metric instead.")


@cli_parse
class DEBUG:
    supervised_value_fn = Proto(False, help="True to turn on supervised grounding for the value function")
    oracle_planning = Proto(False, help="True to turn on oracle planner with ground-truth value function")


class SingleDataset(Dataset):
    def __init__(self, data, target, train=False, ):
        self.data = data
        self.target = target

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)


def map_where(cond, rs_0, rs_1):
    """ map np where to each ouf the output of a and b

    :param cond:
    :param rs_0:
    :param rs_1:
    :return:
    """
    return [np.where(np.broadcast(cond, a), a, b) for a, b in zip(rs_0, rs_1)]


# noinspection NonAsciiCharacters
def train(*, all_images, num_epochs, batch_n, optim_batch_size, ρ, ρ_goal,
          ρ_eval, ρ_eval_goal, H, φ, Φ, inds, N, Φ_target=None, ):
    """

    note: Algorithm Summary
     ✓ Require: planning horizon H=50
     ✓ Require: set of observation sequences S={ τ=x[0:T] }
     ✓ Require: local metric function ϕ(x,x′) ⇒ R+
       ~
     ✓ 1: Initialize global embedding φ(x,x′) ⇒ R+
     ✓ 2: repeat
       3:   sample x0,xg ∈ S as start and goal
       4:   repeat{h=0, h++}
       5:     find set n = { x′s.t.φ(x0,x′)∈N(1,) }
       6:     find x∗ = arg\min_x∈n φ(x, xg)     <~ this is changed
       7:     compute rt = r(x∗, xg)
       8:     add〈x, x∗, rt, xg〉to bufferB
       9:   until r = 0 or h = H
      9.5:  relabel 𝜏 with intermediate goals (every k)
      10:   Sample〈x, x′, r, xg〉from B
      11:   minimize δ = Vφ(x, xg) − r(x, x′, xg) − Vφ(x′, xg)
      12: until convergence

    :param num_epochs:
    :param batch_n:
    :param optim_batch_size:
    :type ρ: function for sampling the state (and goals), the static
      distribution of states.
           ρ(s) := P(s).
    :param ρ_goal: function for sampling from the past goals in the dataset.
                 ρ_goal(goal) := P(goal).
    :type ρ_eval: sampler for evaluation starting points.
    :param H: the planning horizon
    :param φ: the local metric
    :param Φ: the global metric we are trying to learn
    :param Φ: the target value function
    :param N: the neighbor search function
    :return:
    """
    from ml_logger import logger
    import matplotlib.pyplot as plt

    optimizer = optim.Adam(Φ.parameters(), lr=Args.lr)


    # value_fn = lambda x, g: Φ(x, g).cpu().numpy()

    criteria = nn.MSELoss(reduce=False)

    edges = dict(enumerate(inds))
    inputs = []
    targets = []
    distances = dijkstras(edges)
    buffer = PrioritizedReplayBuffer(Args.data_size ** 2, alpha=Args.alpha)
    beta_schedule = LinearSchedule(Args.num_epochs, initial_p=0.4, final_p=1.0)
    for (p, p_), d in distances.items():
        # inputs.append((all_images[p], all_images[p_]))
        # targets.append(d * Args.r_scale)
        buffer.extend(x=all_images[p][None, ...], goal=all_images[p_][None, ...], distance=[d * Args.r_scale])

    fig = plt.figure(figsize=(6, 3), dpi=70)
    plt.title('Distances', fontsize=14)
    print(max(list(distances.values())))
    plt.hist(list(distances.values()), bins=int(max(distances.values())), rwidth=0.9, color="#23aaff", alpha=0.7, histtype='stepfilled')
    plt.xticks(np.arange(1, max(list(distances.values())), 10), range(1, max(list(distances.values())), 10))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig("distances.pdf")

    # dataloader = DataLoader(Dataset(inputs, targets), batch_size=Args.optim_batch_size)
    done, k = np.zeros(batch_n), np.zeros(batch_n)
    steps = 0
    for epoch in range(Args.num_epochs):
        beta = beta_schedule.value(epoch)
        # for data, target in dataloader:
        for i in range(Args.data_size ** 2 // Args.optim_batch_size):
            steps += 1
            paths = buffer.sample(Args.optim_batch_size, beta)
            target = torch.tensor(paths['distance'])
            values = torch.stack(sliced_helper(Φ)(paths['x'], paths['goal'])).squeeze()
            loss = criteria(values, target.to(Args.device).float())
            new_priorities = np.abs(loss.cpu().data.numpy()) + 1e-6
            buffer.update_priorities(paths['inds'], new_priorities)

            loss = torch.mul(torch.Tensor(paths['weights']).to(Args.device), loss).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ############# eval ##################
            # deterministically generate eval version of this
            if steps % 10 == 0:
                x, x_inds = ρ(batch_n)
                xg, xg_inds = ρ_goal(batch_n)
                finished_evals = []
                evals = [x_inds]
                traj = defaultdict(list)
                with torch.no_grad():
                    for plan_step in range(H):  # sampler rollout length
                        ns, ds, inds = N(x_inds, visited=None)  # Size(batch, k)

                        # cost = φ(np.broadcast_to(np.expand_dims(x, 1), ns.shape), ns) + \
                        #        Φ(ns, np.broadcast_to(np.expand_dims(xg, 1), ns.shape))
                        cost = sliced_helper(Φ)(ns, np.expand_dims(xg, 1))
                        # cost = [torch.Tensor([distances[(ind, xg_ind)] for ind in inds[idx]]).to(Args.device) for idx, xg_ind in enumerate(xg_inds)]

                        # greedy
                        _ = np.array([torch.argmin(c).cpu().numpy() for c in cost])

                        x_star = [n[ind] for ind, n in zip(_, ns)]
                        x_star_inds = [n_inds[ind] for ind, n_inds in zip(_, inds)]

                        r = φ(x_star, xg)
                        success = torch.cat([_ <= Args.term_r for _ in r]).squeeze(-1).cpu().numpy()
                        done = success | (k >= H)

                        # todo: use success (saved in traj) to label termination

                        # note: for HER, preserving the rollout structure improves relabel efficiency.
                        traj["x"].append(x)
                        traj["next"].append(x_star)
                        traj["r"].append(r.squeeze(-1).cpu().numpy())
                        traj["goal"].append(xg)
                        traj["done"].append(done)
                        traj['success'].append(success)

                        evals.append(x_star_inds)
                        for idx, d in enumerate(done):
                            if d:
                                finished_evals.append([evals[i][idx] for i in range(len(evals))] + [xg_inds[idx]])

                        logger.store_metrics(loss=loss.cpu().item(), ds=ds, r=r.cpu().numpy(), episode_len=(k + 1)[np.ma.make_mask(done)])

                        assert success.shape == done.shape, "mismatch kills kittens"
                        if done.sum():
                            logger.store_metrics(success_rate=np.sum(success * done) / done.sum())

                        k = np.where(done, 0, k + 1)

                        _x, _x_inds = ρ(batch_n)
                        _xg, _xg_inds = ρ_goal(batch_n)

                        x = np.where(done[:, None, None, None], _x, x_star)
                        x_inds = np.where(done, _x_inds, x_star_inds)
                        xg = np.where(done[:, None, None, None], _xg, xg)
                        xg_inds = np.where(done, _xg_inds, xg_inds)

                    traj = {k: np.array(l) for k, l in traj.items()}

                    logger.log_data(finished_evals, f"finished_evals_{epoch}.pkl")
                    _ = traj['x'][:20, :6, 0].transpose(1, 0, 2, 3)
                    logger.log_images(_.reshape(-1, 64, 64), f"figures/planned/{epoch:04}_plan.png", n_cols=20, n_rows=6)
                    _ = traj['done'][:20, :6].astype(np.uint8).tolist()
                    logger.log_text("\n".join([", ".join([str(item) for item in row]) for row in _]) + "\n",
                                    filename=f"figures/planned/{epoch:04}_done.csv")
                    # log image identity
                    _ = traj['goal'][:20, :6, 0].transpose(1, 0, 2, 3)
                    logger.log_images(_.reshape(-1, 64, 64), f"figures/planned/{epoch:04}_goal.png", n_cols=20, n_rows=6)
                logger.log_metrics_summary(ds="min_max", r="min_max", episode_len="quantile",
                                           key_stats={"debug/supervised_rg": "min_max"},
                                           key_values=dict(epoch=epoch, timesteps=int(epoch * batch_n * H)))


def pairwise_fn(φ, xs, xs_slice, chunk=3, row_chunk=8000):
    from tqdm import tqdm

    return torch.cat([
        torch.cat([
            φ(*torch.broadcast_tensors(_xs.unsqueeze(0), _xs_slice.unsqueeze(1))).squeeze(-1)
            for _xs_slice in tqdm(tslice(xs_slice, chunk=chunk), desc="computing pair-wise distance")
        ])
        for _xs in tslice(xs, chunk=row_chunk)], dim=-1)


def dijkstras(edges):
    visited = defaultdict(set)
    distances = {}
    predecessors = {}

    for src in range(len(edges)):
        unvisited = []

        distances[(src, src)] = 0
        heappush(unvisited, (0, src))
        while unvisited:
            # visit the neighbors
            dist, v = heappop(unvisited)
            if v in visited[src] or v not in edges:
                continue
            visited[src].add(v)

            neighbors = list(edges[v])

            for idx, neighbor in enumerate(neighbors):
                if neighbor not in visited[src]:
                    new_distance = distances[(src, v)] + 1
                    if new_distance < distances.get((src, neighbor), float('inf')):
                        distances[(src, neighbor)] = new_distance
                        heappush(unvisited, (new_distance, neighbor))
                        predecessors[neighbor] = v
    return distances


def build_edges(xs, φ, k=5, slice_index=None, size=None):
    """Factory for get_neighbor functions.

    This version differs from the state-space neighbor factor in that this one
    computes the distance matrix `ds` piece-by-piece. This is because the raw
    images for computing the entire matrix is 740 GB, too large to fit on a single
    card with CUDA.

    :param xs: Size(batch_n, feat_n) the complete set of samples from which we want
      to build the graph.
    :param k: Int(top k inputs) k
    :param slice_index: the map_reduce index for the slice.
    :return: selected neighbors Size(batch_n)
    """
    import torch
    from tqdm import trange
    from ml_logger import logger

    with torch.no_grad():
        if Args.load_neighbors:
            _ = logger.load_pkl(Args.load_neighbors)
            top_ds = _['top']
            inds = _['inds']
        elif Args.load_pairwise_ds:
            with logger.PrefixContext(Args.load_pairwise_ds):
                _ds = np.concatenate([logger.load_pkl(f"chunk_{k:02d}.pkl")[0] for k in trange(20, desc="load")])
            ds = torch.tensor(_ds, dtype=torch.float32)[:size, :size]
            ds[torch.eye(len(ds), dtype=torch.uint8)] = float('inf')
            for k in range(1, 3):  # add true neighbors
                diag = torch.diagflat(torch.ones(len(ds) - k, dtype=torch.uint8), k)
                ds[diag] = 0.5
                diag = torch.diagflat(torch.ones(len(ds) - k, dtype=torch.uint8), -k)
                ds[diag] = 0.5
            # with torch.no_grad():
            #     _ds = torch.tensor(ds)
            #     _ds[torch.eye(ds.shape[0], dtype=torch.uint8)] = float('inf')
            #     _ = torch.topk(_ds, k=24, dim=1, largest=False, sorted=True)
            # top_ds = _[0].numpy()
            # inds = _[1].numpy()

            # # add threshold connection instead.
            full_range = torch.arange(len(ds))
            inds = [None] * len(ds)
            for idx, row in enumerate(ds):
                visited = []
                inds[idx] = full_range[row <= Args.term_r].numpy()
                if len(inds[idx]) == 0:
                    raise Exception('term_r too small')
                    break
                for i in range(1):
                    new_neighbors = [full_range[ds[id] <= Args.term_r].numpy() for id in inds[idx] if id not in visited]
                    if len(new_neighbors) > 0:
                        neighbors = np.concatenate(new_neighbors)
                    else:
                        neighbors = []
                    visited += list(inds[idx])
                    inds[idx] = np.unique(np.concatenate([inds[idx], neighbors]))
                    inds[idx] = inds[idx][inds[idx] != idx]  # remove identity
            inds = np.array(inds)
            top_ds = np.array([row[_] for row, _ in zip(ds.numpy(), inds)])
            logger.log_data(data=dict(top=top_ds, inds=inds), path="top_ds.npy")
        elif Args.load_top_k:
            # _ = logger.load_pkl(Args.load_top_k)[0]
            # top_ds = _['top']
            # inds = _['inds']
            k = 24
            with logger.PrefixContext(Args.load_top_k):
                _ds = np.concatenate([logger.load_pkl(f"chunk_{k:02d}.pkl")[0] for k in trange(20, desc="load")])
            ds = torch.tensor(_ds, dtype=torch.float32)[:size, :size]
            ds[torch.eye(len(ds), dtype=torch.uint8)] = float('inf')
            with torch.no_grad():
                _ds = torch.tensor(ds)[:size, :size]
                _ds[torch.eye(ds.shape[0], dtype=torch.uint8)] = float('inf')
                top_ds, inds = torch.topk(_ds, k=k, dim=1, largest=False, sorted=True)
        else:
            raise Exception("Need to specify pre-computed pairwise distance matrix, or top-k")

    return inds, xs, top_ds


def main(_DEBUG=None, **kwargs, ):
    from ml_logger import logger

    Args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Args.device = torch.device("cpu")

    Args.update(kwargs)
    if _DEBUG is not None:
        DEBUG.update(_DEBUG)

    logger.log_params(Args=vars(Args), DEBUG=vars(DEBUG))

    np.random.seed(Args.seed)
    torch.manual_seed(Args.seed)
    # see https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if True:  # load local metric
        cprint('loading local metric', "yellow", end="... ")

        # hard code args for experiment
        f_local_metric = LocalMetricConvDeep(1, 32).to(Args.device)
        logger.load_module(f_local_metric, Args.load_local_metrics)

        cprint('✔done', 'green')

    if True:  # get rope dataset
        cprint('loading rope dataset', "yellow", end="... ")

        from os.path import expanduser

        rope = np.load(expanduser(Args.data_path))[16:17]
        all_images = np.concatenate(rope / 255).transpose(0, 3, 1, 2).astype(np.float32)[:Args.data_size]
        traj_labels = np.concatenate([np.ones(len(traj)) * i for i, traj in enumerate(rope)])
        print('dataset size', len(all_images))
        cprint('✔done', 'green')

    def sample_obs(batch_n):
        """Sample a batch of `batch_n` observations and indices
         from the the dataset, with *replacement*.

        :param batch_n: (int) the size for the returned batch
        :return: Size(batch_n, 1)
        """
        if Args.single_traj is None:
            low, high = 0, len(all_images)
        else:
            low = np.argmax(traj_labels == Args.single_traj)
            high = np.argmax(traj_labels == Args.single_traj + 1)
        inds = np.random.randint(low, high, size=batch_n)
        # inds = np.array([0] * batch_n)
        return all_images[inds], inds

    def sample_goal_obs(batch_n):
        """Sample a batch of `batch_n` observations and indices
         from the the dataset, with *replacement*.

        :param batch_n: (int) the size for the returned batch
        :return: Size(batch_n, 1)
        """
        if Args.single_traj is None:
            low, high = 0, len(all_images)
        else:
            low = np.argmax(traj_labels == Args.single_traj)
            high = np.argmax(traj_labels == Args.single_traj + 1)
        inds = np.random.randint(low, high, size=batch_n)
        inds = np.array([10] * batch_n)
        return all_images[inds], inds

    def N(x_indices, visited=None):
        """Query the neighbors using the indices

        :param x_indices: Size(m,)
        :return:      x_next,         dstance,           indices
        with shape:  Size(B, k, 2),   Size(B
        """
        nonlocal xs, top_ds, inds

        if visited is not None:
            curr_inds = [[ind for ind in inds[_] if ind not in visited[idx]] for idx, _ in enumerate(x_indices)]
        else:
            curr_inds = [inds[_] for _ in x_indices]
        return zip(*[(xs[curr_inds[idx]], top_ds[_], curr_inds[idx]) for idx, _ in enumerate(x_indices)])

    eval_starts = list(range(0, len(all_images), 10))[:Args.batch_n]

    def eval_obs(batch_n):
        """:param batch_n: (int) the size for the returned batch
        :return: Size(batch_n, 1)
        """
        return all_images[eval_starts], eval_starts

    def eval_obs_goal(batch_n):
        """:param batch_n: (int) the size for the returned batch
        :return: Size(batch_n, 1)
        """
        return all_images[np.array(eval_starts) + 40], np.array(eval_starts) + 40

    inds, xs, top_ds = build_edges(xs=all_images, φ=f_local_metric, k=Args.k, size=len(all_images))

    # xs, xs_inds = sample_obs(10)
    # ns, ds, ns_inds = N(xs_inds)
    # visualize_rope_neighbors(xs, ns, f"figures/neighborhood.png")

    global_metric = GlobalMetricConvDeepL1Norm(1, Args.latent_dim).to(Args.device)
    target_global_metric = GlobalMetricConvDeepL1Norm(1, 32).to(Args.device)
    if not Args.double_dqn:
        target_global_metric.load_state_dict(global_metric.state_dict())

    train(all_images=all_images, num_epochs=Args.num_epochs, batch_n=Args.batch_n,
          optim_batch_size=Args.optim_batch_size,
          ρ=sample_obs,
          ρ_goal=sample_obs,
          ρ_eval=eval_obs,
          ρ_eval_goal=eval_obs_goal,
          H=Args.H,
          φ=torchify(f_local_metric, dtype=torch.float32, input_only=True),
          Φ=torchify(global_metric, dtype=torch.float32, input_only=True),
          inds=inds,
          N=N,
          Φ_target=torchify(target_global_metric, input_only=True)) if Args.target_update else None,


if __name__ == "__main__":
    from ml_logger import logger
    from plan2vec_experiments import instr

    Args.load_local_metrics = "/amyzhang/plan2vec/2019/05-14/plan2vec/local_metric_rope/18.00/33.832397/models/local_metric_002-080.pkl"
    Args.load_pairwise_ds = "/amyzhang/plan2vec/2019/05-19/rope_pairwise/run-12.27.18"
    # Args.load_top_k = "~/fair/pairwise.npy"
    # Args.load_neighbors = "/amyzhang/plan2vec/2019/05-16/plan2vec/plan2vec_rope/09.09/18.306311/top_ds.npy"
    Args.double_dqn = True
    Args.eps_greedy = 0.1
    Args.binary_reward = True

    if False:
        _ = thunk(main, seed=5 * 100)
        _()
    else:
        cprint('Training on cluster', 'green')
        import jaynes

        jaynes.config("learnfair-gpu")
        # jaynes.config("priority-gpu")
        for Args.H in [50]:
            Args.double_dqn = True
            _ = instr(main, seed=5 * 100)
            jaynes.run(_)
        jaynes.listen()
