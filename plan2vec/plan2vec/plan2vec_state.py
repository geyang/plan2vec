from collections import defaultdict
import torch
from more_itertools import flatten
from params_proto.neo_proto import ParamsProto, Proto
from termcolor import cprint
from torch import optim, nn
import numpy as np
from tqdm import trange

from torch_utils import torchify, sliced_helper
from plan2vec.mdp.helpers import make_env
from plan2vec.models.mlp import GlobalMetricFusion, GlobalMetricMlp, GlobalMetricLinearKernel, GlobalMetricKernel, \
    GlobalMetricL2, GlobalMetricAsymmetricL2, LocalMetric
from plan2vec.mdp.replay_buffer import ReplayBuffer
from plan2vec.mdp.sampler import path_gen_fn
from plan2vec.mdp.wrappers.subproc_vec_env import SubprocVecEnv
from plan2vec.plotting.plot_q_value import visualize_q_2d
from plan2vec.plotting.visualize_pairs_2d import visualize_latent_plans, visualize_neighbors
from plan2vec.plotting.maze_world.embedding_state_maze import visualize_embedding_2d, visualize_embedding_3d


class Args(ParamsProto):
    env_id = 'GoalMassDiscreteIdLess-v0'
    start_seed = 0

    # sampling
    n_rollouts = Proto(400, help="The total number of rollouts for the passive dataset")
    num_envs = Proto(20, help="this is used only for sampling a passive dataset from an MDP")
    timesteps = Proto(10, help="the number of timesteps for each rollout")

    latent_dim = Proto(3, help="The latent dimension for the global metric function")

    load_local_metric = Proto("models/local_metric.pkl", help="location to load the weight from")
    # global metric is not set to eval.
    load_global_metric = Proto(None, help="whether to, and where to load the global metric")

    global_metric = "GlobalMetricAsymmetricL2"

    full_cost = Proto(False, help="when planning, whether use the cost to the goal or "
                                  "the full cost including that from current position")
    top_k = Proto(None, help="k neighbors, 30 ~ 200 is usually a good range, depending on the size of the dataset.")
    neighbor_r = Proto(1, help="threshold for selecting neighbors")
    term_r = Proto(1, help="the termination threshold for the local metric score")
    include_traj_neighbors = Proto(True, help="include neighbors in the trajectory in pairwise matrix."
                                              "This allows us to improve the accuracy without losing recall.")
    plan_steps = Proto(1, help="Search for neighbors that are 3 steps away from the current observed state.")

    r_scale = 1 / 20.

    num_epochs = 10
    H = Proto(50, help="planning horizon for the policy")
    gamma = Proto(1, help="discount factor")

    eps_greedy = Proto(0.05, help="epsilon greedy factor. Explore when rn < eps")
    relabel_k = Proto(5, help="the k-interval for the relabeling")
    optim_steps = 6
    optim_batch_size = 32

    target_update = Proto(0.9, help="when less than 1, does soft update. When larger or equal to 1, does hard update.")

    lr = 0.001
    weight_decay = 0.0005

    # evaluation_interval = 10
    visualization_interval = 10
    binary_reward = Proto(False, help="Uses a binary reward if true. Otherwise uses local metric instead.")


class EVAL(ParamsProto):
    eval_grid = [-0.15, 0, 0.15]
    eval_pick_r = Proto(0.04, help="ground-truth radius for finding the nearest sample point as evaluation points.")


class DEBUG(ParamsProto):
    supervised_value_fn = Proto(False, help="True to turn on supervised grounding for the value function")
    oracle_planning = Proto(False, help="True to turn on oracle planner with ground-truth value function")
    oracle_eps_greedy = Proto(False, help="True to turn on eps-greedy with oracle distance metric.")
    random_policy = Proto(False, help="True to run with random policy")
    real_r_distance = Proto(False, help="Overwritten when supervised_value_fn is True. This one toggles "
                                        "whether we use the real distance for the reward.")
    use_true_goals = Proto(True, help="sample goals from the true goals in the trajectory. Note that this"
                                      "is smaller in size, making it easier to over-fit. Default to True.")
    ground_truth_neighbor_r = Proto(False, help="when not zero, use this as a threshold for the ground"
                                                "truth neighbors")
    collect_neighborhood_size = Proto(False, help="flag for collecting the size of the neighborhood being"
                                                  "collected")


def map_where(cond, rs_0, rs_1):
    """ map np where to each ouf the output of a and b

    :param cond:
    :param rs_0:
    :param rs_1:
    :return:
    """
    return [np.where(np.broadcast(cond, a), a, b) for a, b in zip(rs_0, rs_1)]


def find_close(coord, coords, pick_r, ord=2):
    ds = np.linalg.norm(coords - coord, ord=ord, axis=-1)
    ind = np.argmin(ds)
    return (coords[ind], ind) if ds[ind] <= pick_r else None


def numpify(fn):
    return lambda *args: fn(*[np.array(a) for a in args])


def identity(x):
    return x


# noinspection NonAsciiCharacters
def train(*, num_epochs, batch_n, optim_batch_size,
          ρ, ρ_goal,
          # ρ_eval, ρ_eval_goal,
          H, φ, Φ, N, Φ_target=None, ):
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
       5:     find set n = { x′ s.t. φ(x0, x′)∈N(1,) }
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
    :type ρ: function for sampling the state (and goals), the static distribution
             of states.
                 ρ(s) := P(s).
    :param ρ_goal: function for sampling from the past goals in the dataset.
                 ρ_goal(goal) := P(goal).
    # :type ρ_eval: sampler for evaluation starting points.
    # :param ρ_eval_goal: function for evaluation goal points.
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

    buffer = ReplayBuffer(2 * Args.H * Args.num_envs)

    optimizer = optim.Adam(Φ.parameters(), lr=Args.lr, weight_decay=Args.weight_decay)

    done, k = np.zeros(batch_n), np.zeros(batch_n)

    value_fn = lambda x, g: np.array([_.cpu().numpy() for _ in Φ(x, g)])

    criteria = nn.MSELoss(reduction='mean')

    logger.split()
    for epoch in range(num_epochs + 1):
        with torch.no_grad():

            # # this is wrong. The start also need to
            # if epoch % Args.evaluation_interval == 0:
            #     eval_flag = np.ones(batch_n)

            if epoch % Args.visualization_interval == 0:
                cprint('visualizing value map', 'green')
                visualize_q_2d(value_fn, goal_n=7, title=f"Value Map", key=f"figures/value_{epoch:05d}.png")
                # visualize_start_goal(x, xg, f"figures/start_goal_{epoch:05d}.png")
                if Args.latent_dim in [2, 3] and Φ.embed is not None:
                    eval(f"visualize_embedding_{Args.latent_dim}d")(  # also save the pdf version
                        Args.env_id, torchify(Φ.embed, dtype=torch.float),
                        filename=f"figures/embedding/embed_{Args.latent_dim}d_{epoch:04}.pdf")

            # deterministically generate eval version of this
            # x, x_inds = map_where(eval_flag[:, None], ρ_eval(batch_n), ρ(batch_n))
            # xg, xg_inds = map_where(eval_flag[:, None], ρ_eval_goal(batch_n), ρ_goal(batch_n))
            x, x_inds = ρ(batch_n)
            xg, xg_inds = ρ_goal(batch_n)

            # collect past traj here, get those that are finished.
            # collect the goals

            traj = defaultdict(list)
            for plan_step in range(H):  # sampler rollout length
                ns, ds, ns_inds = N(x_inds)  # Size(batch, k)

                if DEBUG.oracle_planning:
                    # true_cost = np.linalg.norm(x - ns, ord=1, axis=-1) +
                    true_cost = [np.linalg.norm(n - np.expand_dims(xg, -2), ord=2, axis=-1) for n, xg in zip(ns, xg)]
                    _ = [np.argmin(c) for c in true_cost]
                elif DEBUG.oracle_eps_greedy:
                    # true_cost = np.linalg.norm(x - ns, ord=1, axis=-1) +
                    true_cost = [np.linalg.norm(n - np.expand_dims(xg, -2), ord=2, axis=-1) for n, xg in zip(ns, xg)]

                    exploration_mask = (np.random.random(Args.num_envs) < Args.eps_greedy) \
                        if Args.eps_greedy else np.zeros(Args.num_envs)

                    _ = [np.random.randint(0, len(c)) if m else np.argmin(c)
                         for c, m in zip(true_cost, exploration_mask)]

                elif DEBUG.random_policy:
                    _ = [np.random.randint(0, len(n)) for n in ns]

                else:
                    if Args.full_cost:
                        cost = [a + b for a, b in zip(Φ(x[:, None], ns), Φ(ns, xg[:, None]))]
                    else:
                        cost = Φ(ns, xg[:, None])

                    if Args.eps_greedy:
                        exploration_mask = np.random.random(Args.num_envs) < Args.eps_greedy
                        _ = [np.random.randint(0, len(c)) if m else c.argmin(0).cpu().item()
                             for c, m in zip(cost, exploration_mask)]
                    else:  # greedy
                        _ = [c.argmin(0).cpu().item() for c in cost]

                # todo: soft-value iteration
                x_star = [n[ind] for ind, n in zip(_, ns)]
                x_star_inds = [n_inds[ind] for ind, n_inds in zip(_, ns_inds)]
                # note: we can use cached distance instead, look up via sample index.
                r = φ(x_star, xg).squeeze(-1)
                success = torch.stack([_ <= Args.term_r for _ in r]).cpu().numpy()
                r = r.cpu().numpy()
                done = success | (k >= H - 1)

                # if epoch % Args.visualization_interval == 0:
                #     visualize_one_step_plans(x[0], xg[0], ns[0], x_star[0],
                #                              f"figures/planning/plan_{epoch:05d}/step_{plan_step:02d}.png")

                # note: for HER, preserving the rollout structure improves relabel efficiency.
                traj["x"].append(x)
                traj["next"].append(x_star)
                traj["r"].append(r)
                traj["goal"].append(xg)
                traj["done"].append(done)
                traj['success'].append(success)
                # traj['is_eval'].append(eval_flag)  # do NOT use if it is eval!

                # if epoch % Args.evaluation_interval == 0:
                #     eval_flag = np.where(done, 1, eval_flag)
                # else:
                #     eval_flag = np.where(done, 0, eval_flag)

                logger.store_metrics(ds=ds, r=r, episode_len=(k + 1)[np.ma.make_mask(done)])
                if done.sum():
                    assert success.shape == done.shape, "mismatch kills kittens"
                    logger.store_metrics(success_rate=np.sum(success * done) / done.sum())

                # assert eval_flag.shape == success.shape, "defensive programming."
                # _done = (1 - eval_flag) * done
                # if _done.sum():
                # _done = eval_flag * done
                # if _done.sum():
                #     logger.store_metrics(metrics={"eval/success_rate": np.sum(success * _done) / _done.sum()})

                k = np.where(done, 0, k + 1)

                _x, _x_inds = ρ(batch_n)
                _xg, _xg_inds = ρ_goal(batch_n)

                x = np.where(done[:, None], _x, x_star)
                x_inds = np.where(done, _x_inds, x_star_inds)
                xg = np.where(done[:, None], _xg, xg)
                xg_inds = np.where(done, _xg_inds, xg_inds)

            traj = {k: np.array(l) for k, l in traj.items()}

            # done: need to preserve rollout structure
            # eval_mask = np.ma.make_mask(traj['is_eval'].reshape(-1))
            # buffer.extend(**{k: v.reshape(-1, *v.shape[2:])[eval_mask] for k, v in traj.items() if k != 'done'})
            buffer.extend(**{k: v.reshape(-1, *v.shape[2:]) for k, v in traj.items() if k != 'done'})
            # done: Hindsight Experience Relabel
            # done: do not re-label if it is eval.
            if Args.relabel_k:
                # todo: need to add eval to this one as well.
                # for goals, goal_is_eval in zip(traj['next'][::Args.relabel_k], traj['is_eval'][::Args.relabel_k]):
                for goals in traj['next'][::Args.relabel_k]:
                    new_goals = np.tile(goals, [H, 1, 1])
                    new_r = φ(traj['next'], new_goals).squeeze(-1).cpu().numpy()

                    # is_eval = goal_is_eval.astype(bool) | traj['is_eval'].astype(bool)
                    buffer.extend(**{
                        # k: v.reshape(-1, *v.shape[2:])[~is_eval.reshape(-1)]
                        k: v.reshape(-1, *v.shape[2:])
                        for k, v in dict(
                            x=traj['x'],
                            next=traj['next'],
                            r=new_r,
                            goal=new_goals,
                            success=new_r < Args.term_r,
                            # is_eval=np.zeros_like(new_r)
                        ).items()
                    })

            # todo: need re-write
            # # note: you can find the same epoch by searching
            # if epoch % Args.visualization_interval == 0:
            #     cprint('visualize trajectories...', 'green')
            #     visualize_latent_plans(traj['x'][:, :1],
            #                            traj['goal'][:, :1],
            #                            traj['done'][:, :1],
            #                            f'figures/path/train_{epoch:05d}.png')

        for i in range(Args.optim_steps):
            paths = buffer.sample(optim_batch_size)
            r = torch.tensor(paths['r'], device=Args.device)
            success = torch.tensor(paths['success'].astype(float), dtype=torch.float)

            # todo: add target value function here.
            values = torch.stack(Φ(paths['x'], paths['goal'])).squeeze()

            with torch.no_grad():
                if DEBUG.supervised_value_fn:
                    r_g = np.linalg.norm(paths['x'] - paths['goal'], ord=2, axis=-1)
                    target_values = torch.Tensor(r_g)
                    logger.store_metrics(metrics={"debug/supervised_rg": r_g})
                elif DEBUG.real_r_distance:
                    mask = 1. - success
                    r_d = np.linalg.norm(paths['x'] - paths['next'], ord=2, axis=-1)
                    logger.store_metrics(metrics={"debug/real_r_d": r_d})
                    target_values = torch.Tensor(r_d) \
                                    + Args.gamma * mask * Φ_target(paths['next'], paths['goal']).squeeze()
                elif Args.binary_reward:
                    mask = 1. - success
                    target_values = mask / Args.r_scale \
                                    + Args.gamma * mask * Φ_target(paths['next'], paths['goal']).squeeze()
                else:
                    mask = 1. - success
                    target_values = r / Args.r_scale \
                                    + Args.gamma * mask * Φ_target(paths['next'], paths['goal']).squeeze()

            true_delta = torch.abs(values - target_values).mean().detach().numpy()
            # δ = F.smooth_l1_loss(values, target_values, reduce='mean').squeeze()
            assert values.shape == target_values.shape, \
                f"broadcasting kills kittens. {values.shape}, {target_values.shape}"
            δ = criteria(values, target_values).squeeze()
            logger.store_metrics(loss=δ.detach().cpu().item(),
                                 true_delta=true_delta.item(),
                                 value=values.detach().cpu().mean().item(),
                                 metrics={"buffer/r": paths['r'].mean()})

            optimizer.zero_grad()
            δ.backward()
            optimizer.step()

            # done: update target value function
            if not Args.target_update:
                continue
            elif Args.target_update < 1:  # <-- soft target update
                Φ_params = Φ.state_dict()
                with torch.no_grad():
                    for name, param in Φ_target.state_dict().items():
                        param.copy_((1 - Args.target_update) * Φ_params[name] + Args.target_update * param)
            elif int(epoch * Args.optim_steps + i) % Args.target_update == 0:  # <-- hard target update
                Φ_target.load_state_dict(Φ.state_dict())

        logger.log_metrics_summary(ds="min_max", r="min_max", episode_len="quantile",
                                   loss="mean", true_delta="mean", value="mean",
                                   key_stats={"debug/supervised_rg": "min_max"},
                                   key_values=dict(epoch=epoch,
                                                   dt_epoch=logger.split(),
                                                   timesteps=int(epoch * batch_n * H)))

        # done: clear buffer to learn with only on-policy samples
        # todo: keep filling the buffer, and learn from recent experiences.
        # buffer.clear()


def main(deps, **kwargs, ):
    from ml_logger import logger

    print(deps)

    # Args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Args.device = torch.device("cpu")

    Args._update(deps, **kwargs)
    DEBUG._update(deps)

    logger.log_params(Args=vars(Args), DEBUG=vars(DEBUG))

    np.random.seed(Args.start_seed)
    torch.manual_seed(Args.start_seed)
    # see https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if True:  # load local metric
        cprint('loading local metric', "green")

        f_local_metric = LocalMetric(2, 50).to(Args.device)
        logger.load_module(f_local_metric, Args.load_local_metric, tries=10)
        f_local_metric.eval()  # Not important for state space, but critical for image.


    if True:  # get training dataset
        cprint('getting training dataset', "green")

        envs = SubprocVecEnv([make_env(Args.env_id, Args.start_seed + i) for i in range(Args.num_envs)])
        logger.log_params(env=envs.spec._kwargs)

        memory = ReplayBuffer(Args.n_rollouts * Args.timesteps)

        random_pi = lambda ob, goal, *_: np.random.randint(0, 8, size=[len(ob)])
        random_path_gen = path_gen_fn(envs, random_pi, "x", "goal")
        next(random_path_gen)

        step_inds = np.arange(Args.timesteps)[..., None] * np.ones([1, Args.num_envs])

        for i in trange(Args.n_rollouts // Args.num_envs):
            paths = random_path_gen.send(Args.timesteps)

            traj_inds = np.arange(i * Args.num_envs, (i + 1) * Args.num_envs)[None, ...] * np.ones([Args.timesteps, 1])

            l = len(memory)
            memory.extend(
                s=paths['obs']['x'].reshape(-1, 2).astype(np.float32),
                s_=paths['next']['x'].reshape(-1, 2).astype(np.float32),
                goal=paths['obs']['goal'].reshape(-1, 2).astype(np.float32),
                traj=traj_inds.reshape(-1),
                step=step_inds.reshape(-1),
                ind=list(range(l, l + Args.num_envs * Args.timesteps)),
            )

        # logger.save_data(memory.buffer, 'data/path_sample.pkl')
        # memory.buffer, = logger.load_data('data/path_sample.pkl')

    def sample_obs(batch_n):
        d = memory.sample(batch_n)
        return d['s'], d['ind']

    def sample_goal(batch_n):
        d = memory.sample(batch_n)
        return d['goal'], d['ind']

    eval_starts = [[a, b] for a in EVAL.eval_grid for b in EVAL.eval_grid]
    eval_goals = [[a, b] for a in EVAL.eval_grid for b in EVAL.eval_grid]

    def eval_factory(coords):
        _ = numpify(find_close)
        nearest_samples, nearest_ids = \
            zip(*filter(identity, [_(c, memory.buffer['s'], EVAL.eval_pick_r) for c in coords]))
        pointer = 0

        # always use obs instead of goals too
        def eval_obs(batch_n):
            nonlocal pointer
            _ = [(p + pointer) % len(nearest_samples) for p in range(batch_n)]
            pointer += batch_n
            return np.array(nearest_samples)[_], np.array(nearest_ids)[_]

        return eval_obs

    if DEBUG.ground_truth_neighbor_r:
        N = ground_truth_neighbor_factory(xs=memory['s'], r=DEBUG.ground_truth_neighbor_r)
    else:
        n_fn = neighbor_factory(xs=memory['s'], traj=memory['traj'], step=memory['step'],
                                φ=torchify(f_local_metric, input_only=True, dtype=torch.float32),
                                top_k=Args.top_k, step_threshold=Args.neighbor_r,
                                include_traj_neighbors=Args.include_traj_neighbors)
        N = lambda x_inds: n_fn(x_inds, Args.plan_steps)

    xs, xs_inds = sample_obs(10)
    ns, ds, ns_inds = N(xs_inds)
    visualize_neighbors(xs, ns, f"figures/neighborhood.png")

    if DEBUG.collect_neighborhood_size:
        xs, xs_inds = sample_obs(100)
        ns, ds, ns_inds = N(xs_inds)
        logger.log_data([len(n) for n in ns], path="neighborhood_size.pkl")

        logger.print('logging the neighborhood size data')

    GlobalMetric = globals().get(Args.global_metric)

    global_metric = GlobalMetric(2, Args.latent_dim).to(Args.device)
    logger.log_text(str(global_metric), filename="models/global_metric.txt", silent=True)

    if Args.load_global_metric:
        logger.load_module(global_metric, Args.load_global_metric)
        # note: not setting global_metric to eval
        # global_metric.eval()  # Not important for state space, but critical for image.

    target_global_metric = GlobalMetric(2, Args.latent_dim).to(Args.device)
    target_global_metric.load_state_dict(global_metric.state_dict())

    train(num_epochs=Args.num_epochs, batch_n=Args.num_envs,
          optim_batch_size=Args.optim_batch_size,
          ρ=sample_obs,
          ρ_goal=sample_goal if DEBUG.use_true_goals else sample_obs,
          # ρ_eval=eval_factory(eval_starts),
          # ρ_eval_goal=eval_factory(eval_goals),
          H=Args.H,
          φ=torchify(f_local_metric, input_only=True, dtype=torch.float32),
          Φ=sliced_helper(torchify(global_metric, input_only=True, dtype=torch.float32)),
          N=N,
          Φ_target=torchify(target_global_metric, input_only=True,
                            dtype=torch.float32) if Args.target_update else None, )


def ground_truth_neighbor_factory(xs, r=0.1, soft=False):
    """Factory for ground-truth neighbors within distance r.

    :param xs: samples
    :param r: radius for the neighborhood
    :param soft: Not used
    :return: neighbors, distances, neighbor indices (in the dataset)
    """

    # implement this in numpy instead of pytorch is okay.
    _ = torch.tensor(xs)
    pairwise_ds = torch.norm(_[:, None, :] - _[None, :, :], dim=-1)
    data_size = len(pairwise_ds)
    # 1. set diag to inf
    pairwise_ds[torch.eye(data_size, dtype=torch.uint8)] = float("inf")
    # 2. get indices of filtered
    full_range = torch.arange(data_size)
    top_inds = [full_range[row <= r] for row in pairwise_ds]
    top_ds = [ds[inds] for ds, inds in zip(pairwise_ds, top_inds)]

    # too big, do not log.
    # from ml_logger import logger
    # logger.log_image(pairwise_ds, f"figures/debug/ground_truth_pairwise_ds.png")

    def N(x_indices):
        nonlocal xs, top_ds, top_inds
        # around a pytorch bug: https://github.com/pytorch/pytorch/issues/20697
        return zip(*[(xs[top_inds[_].tolist()], top_ds[_], top_inds[_]) for _ in x_indices])

    return N


def neighbor_factory(xs, traj, step, φ, top_k=None, step_threshold=None, soft=False, include_traj_neighbors=True):
    """Factory for get_neighbor functions.

    :param xs: Size(batch_n, feat_n) the complete set of samples from which we want
      to build the graph.
    :type traj: Size(batch_n) the label for which trajectory these sample come from
    :type step: Size(batch_n) the label for the step index within the original trajectory
    :param φ: the local metric function we use to compute the 1-step neighbors
    :param top_k: Int(top k inputs) k for top-k neighbors
    :param step_threshold: The threshold for each planning step
    :param include_traj_neighbors: includes ground-truth neighbors from each trajectory.
    :return: a neighbor function.
    """

    n, *feat_n = xs.shape
    a = np.broadcast_to(xs[None, ...], [n, n, *feat_n])
    b = np.broadcast_to(xs[:, None, ...], [n, n, *feat_n])
    # this should really be the local_metric function
    # ds = np.linalg.norm(a - b, ord=None, axis=-1, keepdims=False)

    with torch.no_grad():

        ds = φ(a, b).squeeze(-1)  # Size(n, n)

        # mask self, preserve indices
        ds[torch.eye(n, dtype=torch.uint8)] = float('inf')

        # add true true trajectory neighbors
        if include_traj_neighbors:
            # use int type to allow negative numbers
            data_size = len(ds)
            step_m = torch.tensor(step)[:, None].repeat([1, data_size])
            traj_m = torch.tensor(traj)[:, None].repeat([1, data_size])
            neighbor_mask = (torch.eq(traj_m, traj_m.transpose(0, 1))
                             & (torch.abs(step_m - step_m.transpose(0, 1)) == 1))
            ds[neighbor_mask] = step_threshold
            from ml_logger import logger
            logger.log_image(ds.cpu().numpy()[:100, :100],
                             f"figures/debug/ds_pairwise_with_in_trajectory_neighbors.png",
                             cmap='viridis', normalize=[None, 2.5])

        if top_k:
            top_ds, top_inds = ds.topk(top_k, dim=-1, largest=False, sorted=True)
            top_ds = top_ds.cpu().numpy()
            top_inds = top_inds.cpu().numpy()
        else:
            # # add threshold connection instead.
            full_range = torch.arange(len(ds))
            top_inds = [full_range[row <= step_threshold].cpu().numpy() for row in ds]
            top_ds = [row[_].cpu().numpy() for row, _ in zip(ds, top_inds)]

    # check for samples with no neighbors
    for i, inds in enumerate(top_inds):
        if len(inds) == 0:
            raise RuntimeError("Point has no neighbor! Is your term_r too stringent? Check "
                               "the visualization under figures/<env-name>_score_vs_gt.png")

    # make sure hashing works
    top_inds = [inds.tolist() for inds in top_inds]

    if soft:  # soft-sample step:
        raise NotImplementedError()

    def N_ind_only(x_indices):
        nonlocal top_inds
        return [top_inds[_] for _ in x_indices]

    def N_multistep(x_inds, k=1):
        nonlocal xs, top_ds, top_inds
        sofar = [{ind} for ind in x_inds]
        for i in range(k):
            new = [set(flatten(N_ind_only(inds))).difference(inds) for inds in sofar]
            sofar = [old.union(n) for old, n in zip(sofar, new)]

        # remove the point itself.
        ns_inds = [_.difference({ind}) for ind, _ in zip(x_inds, sofar)]
        # watch out for a pytorch bug: https://github.com/pytorch/pytorch/issues/20697
        return zip(*[(xs[_], ds[x_ind].cpu().numpy()[_], _) for x_ind, _ in zip(x_inds, map(list, ns_inds))])

    return N_multistep


if __name__ == "__main__":
    from ml_logger import logger
    from plan2vec_experiments import instr

    _ = instr(main, seed=5 * 100)

    if True:
        _()
    elif False:
        cprint('Training on cluster', 'green')
        import jaynes

        jaynes.config("learnfair")
        jaynes.run(_)
        jaynes.listen()
