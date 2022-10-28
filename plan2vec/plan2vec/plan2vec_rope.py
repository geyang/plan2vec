from collections import defaultdict
import torch
from termcolor import cprint
from torch import optim, nn
import numpy as np
from params_proto import cli_parse, Proto
from heapq import heappush, heappop
from baselines.common.schedules import LinearSchedule


from torch_utils import torchify, tslice, sliced_helper
from plan2vec.models.convnets import LocalMetricConvLarge, LocalMetricConvDeep, GlobalMetricConvDeepKernel, GlobalMetricConvDeepL1
from plan2vec.mdp.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

@cli_parse
class Args:
    seed = 0
    data_size = Proto(3000, help="number of images to use")

    # Required data paths
    data_path = Proto("~/fair/new_rope_dataset/data/new_rope.npy", help="path to the data file")
    load_local_metric = Proto("models/local_metric.pkl", help="location to load the weight from")
    load_pairwise_ds = Proto(None, help="pre-computed distance between all samples, using the local metric")
    load_top_k = Proto(None, help="pre-computed top-k, using the local metric")
    load_neighbors = Proto(None, help="pre-computed neighbors, using the local metric")
    single_traj = Proto(None, help="train with only a single trajectory", dtype=int)
    double_dqn = Proto(False, dtype=bool, help="use Double DQN")

    term_r = Proto(1.3, help="the threshold value for a sample being considered a `neighbor`.")

    num_epochs = 5000
    batch_n = 20
    H = Proto(200, help="planning horizon for the policy")
    k = Proto(24, help="k neighbors")
    gamma = Proto(1, help="discount factor")

    r_scale = Proto(1 / 20, help="scaling factor for the reward")
    max_eps_greedy = Proto(0.1, help="epsilon greedy factor. Explore when rn < eps")
    min_eps_greedy = Proto(0.1)

    relabel_k = Proto(5, help="the k-interval for the relabeling")
    optim_steps = 5
    optim_batch_size = 128

    target_update = Proto(0.9, help="when less than 1, does soft update. When larger or equal to 1, does hard update.")

    lr = 0.0001

    k_fold = Proto(10, help="The k-fold validation for evaluating the planning module.")
    evaluation_interval = 10
    visualization_interval = 1
    binary_reward = Proto(False, help="Uses a binary reward if true. Otherwise uses local metric instead.")


@cli_parse
class DEBUG:
    supervised_value_fn = Proto(False, help="True to turn on supervised grounding for the value function")
    oracle_planning = Proto(False, help="True to turn on oracle planner with ground-truth value function")


def map_where(cond, rs_0, rs_1):
    """ map np where to each ouf the output of a and b

    :param cond:
    :param rs_0:
    :param rs_1:
    :return:
    """
    return [np.where(np.broadcast(cond, a), a, b) for a, b in zip(rs_0, rs_1)]


# noinspection NonAsciiCharacters
def train(*, num_epochs, batch_n, optim_batch_size, ρ, ρ_goal,
          ρ_eval, ρ_eval_goal, H, φ, Φ, N, Φ_target=None, ):
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

    if Φ_target is None:
        Φ_target = Φ

    # buffer = ReplayBuffer(2 * Args.H * Args.batch_n)
    buffer = PrioritizedReplayBuffer(2 * Args.H * Args.batch_n, alpha=0.6)
    beta_schedule = LinearSchedule(Args.num_epochs, initial_p=0.4, final_p=1.0)

    eps_schedule = LinearSchedule(100, initial_p=Args.max_eps_greedy, final_p=Args.min_eps_greedy)

    if Args.double_dqn:
        optimizer = optim.Adam(list(Φ.parameters()) + list(Φ_target.parameters()), lr=Args.lr)
    else:
        optimizer = optim.Adam(Φ.parameters(), lr=Args.lr)

    done, k = np.zeros(batch_n), np.zeros(batch_n)

    # value_fn = lambda x, g: Φ(x, g).cpu().numpy()

    criteria = nn.MSELoss(reduce=False)
    for epoch in range(num_epochs + 1):
        eps_greedy = eps_schedule.value(epoch)
        beta = beta_schedule.value(epoch)
        with torch.no_grad():
            # todo: visualize the value map here
            # todo: visualize the planned trajectories
            if epoch % Args.evaluation_interval == 0:
                eval_flag = np.ones(batch_n)

            # deterministically generate eval version of this
            x, x_inds = map_where(eval_flag[:, None, None, None], ρ_eval(batch_n), ρ(batch_n))
            xg, xg_inds = map_where(eval_flag[:, None, None, None], ρ_eval_goal(batch_n), ρ_goal(batch_n))
            visited = [[_] for _ in x_inds]
            if epoch % Args.evaluation_interval == 0:
                finished_evals = []
                evals = [x_inds]
            traj = defaultdict(list)
            for plan_step in range(H):  # sampler rollout length
                ns, ds, inds = N(x_inds, visited=None)  # Size(batch, k)
                if DEBUG.oracle_planning:
                    # true_cost = np.linalg.norm(x - ns, ord=1, axis=-1) + \
                    true_cost = np.linalg.norm(ns - np.expand_dims(xg, -2), ord=2, axis=-1)
                    cost = torch.Tensor(true_cost).to(Args.device)
                    _ = torch.argmin(cost.squeeze(dim=-1), dim=1).cpu().numpy()
                else:
                    exploration_mask = torch.Tensor(np.random.random(Args.batch_n)) < eps_greedy \
                        if eps_greedy else np.zeros(Args.batch_n)
                    exploration_mask *= (1 - torch.ByteTensor(eval_flag))

                    # cost = φ(np.broadcast_to(np.expand_dims(x, 1), ns.shape), ns) + \
                    #        Φ(ns, np.broadcast_to(np.expand_dims(xg, 1), ns.shape))
                    cost = sliced_helper(Φ)(ns, np.expand_dims(xg, 1))

                    # greedy
                    _greedy = np.array([torch.argmin(c).cpu().numpy() for c in cost])
                    _random = np.array([torch.randint(0, len(c), size=tuple()).numpy() for c in cost])
                    assert _random.shape == _greedy.shape, \
                        f"shape mismatch also kills kittens. {_random.shape}:{_greedy.shape}"
                    _ = np.where(exploration_mask, _random, _greedy)

                x_star = [n[ind] for ind, n in zip(_, ns)]
                x_star_inds = [n_inds[ind] for ind, n_inds in zip(_, inds)]
                for _, x_star_ind in enumerate(x_star_inds):
                    visited[_].append(x_star_ind)
                r = φ(x_star, xg)
                success = torch.cat([_ <= Args.term_r for _ in r]).squeeze(-1).cpu().numpy()
                done = success | (k >= H - 1)

                # todo: use success (saved in traj) to label termination

                # note: for HER, preserving the rollout structure improves relabel efficiency.
                traj["x"].append(x)
                traj["next"].append(x_star)
                traj["r"].append(r.squeeze(-1).cpu().numpy())
                traj["goal"].append(xg)
                traj["done"].append(done)
                traj['success'].append(success)
                traj['is_eval'].append(eval_flag)  # do NOT use if it is eval!

                if epoch % Args.evaluation_interval == 0:
                    evals.append(x_star_inds)
                    for idx, d in enumerate(done):
                        if d:
                            finished_evals.append([evals[i][idx] for i in range(len(evals))] + [xg_inds[idx]])
                    eval_flag = np.where(done, 1, eval_flag)
                else:
                    eval_flag = np.where(done, 0, eval_flag)

                logger.store_metrics(ds=ds, r=r.cpu().numpy(), episode_len=(k + 1)[np.ma.make_mask(done)])

                assert eval_flag.shape == success.shape, "defensive programming."
                assert success.shape == done.shape, "mismatch kills kittens"
                _done = (1 - eval_flag) * done
                if _done.sum():
                    logger.store_metrics(success_rate=np.sum(success * _done) / _done.sum(), eps_greedy=eps_greedy)
                _done = eval_flag * done
                if _done.sum():
                    logger.store_metrics(metrics={"eval/success_rate": np.sum(success * _done) / _done.sum()})

                k = np.where(done, 0, k + 1)

                _x, _x_inds = ρ(batch_n)
                _xg, _xg_inds = ρ_goal(batch_n)

                x = np.where(done[:, None, None, None], _x, x_star)
                x_inds = np.where(done, _x_inds, x_star_inds)
                xg = np.where(done[:, None, None, None], _xg, xg)
                xg_inds = np.where(done, _xg_inds, xg_inds)
                for idx, d in enumerate(done):
                    if d:
                        visited[idx] = [x_inds[idx]]

            traj = {k: np.array(l) for k, l in traj.items()}

            if epoch % Args.evaluation_interval == 0:
                logger.log_data(finished_evals, f"finished_evals_{epoch}.pkl")

            # 1. get traj.shape
            # 2. log on interval
            # 3. add lable for done on traj.
            #    traj['x'].shape == Size(H, batch_n, 1, 64, 64)
            if epoch % Args.visualization_interval == 0:
                _ = traj['x'][:20, :6, 0].transpose(1, 0, 2, 3)
                logger.log_images(_.reshape(-1, 64, 64), f"figures/planned/{epoch:04}_plan.png", n_cols=20, n_rows=6)
                _ = traj['done'][:20, :6].astype(np.uint8).tolist()
                logger.log_text("\n".join([", ".join([str(item) for item in row]) for row in _]) + "\n",
                                filename=f"figures/planned/{epoch:04}_done.csv")
                # log image identity
                _ = traj['goal'][:20, :6, 0].transpose(1, 0, 2, 3)
                logger.log_images(_.reshape(-1, 64, 64), f"figures/planned/{epoch:04}_goal.png", n_cols=20, n_rows=6)

            # done: need to preserve rollout structure
            eval_mask = np.ma.make_mask(traj['is_eval'].reshape(-1))
            buffer.extend(**{k: v.reshape(-1, *v.shape[2:])[eval_mask] for k, v in traj.items() if k != 'done'})

            # done: Hindsight Experience Relabel
            if Args.relabel_k:
                for goals, goal_is_eval in zip(traj['next'][::Args.relabel_k], traj['is_eval'][::Args.relabel_k]):
                    new_goals = np.tile(goals, [H, 1, 1, 1, 1])
                    new_r = φ(traj['next'], new_goals).squeeze(-1).cpu().numpy()

                    is_eval = goal_is_eval.astype(bool) | traj['is_eval'].astype(bool)
                    buffer.extend(**{
                        k: v.reshape(-1, *v.shape[2:])
                        for k, v in dict(
                            x=traj['x'],
                            next=traj['next'],
                            r=new_r,
                            goal=new_goals,
                            success=new_r < Args.term_r,
                            is_eval=np.zeros_like(new_r)
                        ).items()
                    })

        for i in range(Args.optim_steps):
            if Args.double_dqn:
                coin_flip = np.random.random()
                if coin_flip > 0.5:
                    Φ_target, Φ = Φ, Φ_target  # swap the two nets
                for p in Φ.parameters():
                    p.requires_grad = True
                for p in Φ_target.parameters():
                    p.requires_grad = False
            paths = buffer.sample(optim_batch_size, beta)
            r = torch.Tensor(paths['r']).to(Args.device)
            success = torch.tensor(paths['success'].astype(float), dtype=torch.float).to(Args.device)

            # done: add target value function.
            values = torch.stack(sliced_helper(Φ)(paths['x'], paths['goal'])).squeeze()

            # todo: use success (saved in traj) to label termination
            with torch.no_grad():
                if Args.binary_reward:
                    mask = 1. - success
                    target_values = mask / Args.r_scale \
                                    + Args.gamma * mask * Φ_target(paths['next'], paths['goal']).squeeze()
                else:
                    mask = 1. - success
                    target_values = r / Args.r_scale \
                                    + Args.gamma * mask * Φ_target(paths['next'], paths['goal']).squeeze()

            true_delta = torch.abs(values - target_values).mean().detach().cpu().numpy()
            # δ = F.smooth_l1_loss(values, target_values, reduce='mean').squeeze()
            assert values.shape == target_values.shape, \
                f"broadcasting kills kittens. {values.shape}, {target_values.shape}"
            δ = criteria(values, target_values).squeeze()
            new_priorities = np.abs(δ.cpu().data.numpy()) + 1e-6
            buffer.update_priorities(paths['inds'], new_priorities)

            δ = torch.mul(torch.Tensor(paths['weights']).to(Args.device), δ).mean()
            logger.store_metrics(loss=δ.detach().cpu().item(),
                                 true_delta=true_delta.item(),
                                 value=values.detach().cpu().mean().item(),
                                 min_value=values.detach().cpu().min().item(),
                                 max_value=values.detach().cpu().max().item(),
                                 metrics={"buffer/r": paths['r'].mean()})

            optimizer.zero_grad()
            δ.backward()
            optimizer.step()

            # done: update target value function
            if not Args.double_dqn:
                if not Args.target_update:
                    continue
                elif Args.target_update < 1:  # <-- soft target update
                    Φ_params = Φ.state_dict()
                    with torch.no_grad():
                        for name, param in Φ_target.state_dict().items():
                            param.copy_((1 - Args.target_update) * Φ_params[name] + Args.target_update * param)
                elif (epoch * Args.optim_steps + _) % Args.target_update == 0:  # <-- hard target update
                    Φ_target.load_state_dict(Φ.state_dict())

        logger.log_metrics_summary(ds="min_max", r="min_max", episode_len="quantile",
                                   key_stats={"debug/supervised_rg": "min_max"},
                                   key_values=dict(epoch=epoch, timesteps=int(epoch * batch_n * H)))

        # todo: clear buffer to learn with only on-policy samples
        # buffer.clear()


def pairwise_fn(φ, xs, xs_slice, chunk=3, row_chunk=8000):
    from tqdm import tqdm

    return torch.cat([
        torch.cat([
            φ(*torch.broadcast_tensors(_xs.unsqueeze(0), _xs_slice.unsqueeze(1))).squeeze(-1)
            for _xs_slice in tqdm(tslice(xs_slice, chunk=chunk), desc="computing pair-wise distance")
        ])
        for _xs in tslice(xs, chunk=row_chunk)], dim=-1)


def dijkstras(src, dest, edges):
    visited = set()
    unvisited = []
    distances = {}
    predecessors = {}

    distances[src] = 1
    heappush(unvisited, (1, src))

    while unvisited:
        # visit the neighbors
        dist, v = heappop(unvisited)
        if v in visited or v not in edges:
            continue
        visited.add(v)
        if v == dest:
            # We build the shortest path and display it
            path = []
            pred = v
            while pred != None:
                path.append(pred)
                pred = predecessors.get(pred, None)
            if len(path) == 1:
                print('foo', src == dest)
                print(src)
                print(dest)
                return None, None
            return path[::-1], 1 / dist

        neighbors = list(edges[v])

        for idx, neighbor in enumerate(neighbors):
            if neighbor not in visited:
                new_distance = distances[v]
                if new_distance < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_distance
                    heappush(unvisited, (new_distance, neighbor))
                    predecessors[neighbor] = v

    # couldn't find a path
    return None, None


def neighbor_factory(xs, φ, k=5, slice_index=None, size=None):
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

    return N


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

        rope = np.load(expanduser(Args.data_path), allow_pickle=True)[16:17]
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

    eval_starts = list(range(0, Args.data_size, 2))[:Args.batch_n]

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

    N = neighbor_factory(xs=all_images, φ=f_local_metric, k=Args.k, size=len(all_images))

    # xs, xs_inds = sample_obs(10)
    # ns, ds, ns_inds = N(xs_inds)
    # visualize_rope_neighbors(xs, ns, f"figures/neighborhood.png")

    global_metric = GlobalMetricConvDeepL1(1, 32).to(Args.device)
    target_global_metric = GlobalMetricConvDeepL1(1, 32).to(Args.device)
    if not Args.double_dqn:
        target_global_metric.load_state_dict(global_metric.state_dict())

    train(num_epochs=Args.num_epochs, batch_n=Args.batch_n,
          optim_batch_size=Args.optim_batch_size,
          ρ=sample_obs,
          ρ_goal=sample_obs,
          ρ_eval=eval_obs,
          ρ_eval_goal=eval_obs_goal,
          H=Args.H,
          φ=torchify(f_local_metric, dtype=torch.float32, input_only=True),
          Φ=torchify(global_metric, dtype=torch.float32, input_only=True),
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
        for Args.H in [100]:
            Args.double_dqn = False
            _ = instr(main, seed=5 * 100)
            jaynes.run(_)
        jaynes.listen()
