from collections import defaultdict
import torch
from more_itertools import flatten
from termcolor import cprint
from torch import optim, nn
import numpy as np
from params_proto import cli_parse, Proto

from torch_utils import torchify, tslice, sliced_helper
from plan2vec.mdp.replay_buffer import ReplayBuffer

from plan2vec.plotting.streetlearn.embedding_image_streetlearn \
    import cache_images, visualize_embedding_2d_image, visualize_embedding_3d_image

from plan2vec.plotting.streetlearn.neighbors import visualize_neighbors_r

### choices for the global model
from plan2vec.models.convnets \
    import LocalMetricConvDeep, GlobalMetricConvL2_s1, GlobalMetricConvDeepL2, GlobalMetricConvDeepL2_wide
from plan2vec.models.resnet import ResNet18L2, ResNet18Stacked

assert [LocalMetricConvDeep]
assert [GlobalMetricConvL2_s1, GlobalMetricConvDeepL2, GlobalMetricConvDeepL2_wide, ResNet18L2, ResNet18Stacked]


@cli_parse
class Args:
    env_id = 'manhattan-tiny'
    seed = 0

    data_path = Proto(f"~/fair/streetlearn/processed-data/{env_id}",
                      help="path to the processed streetlearn dataset")
    street_view_size = Proto((64, 64), help="image size for the dataset", dtype=tuple)
    street_view_mode = Proto("omni-gray", help="OneOf[`omni-gray`, `ombi-rgb`]")

    Lp = Proto(2, help="the power for the L_p metric")

    lng_lat_correction = Proto(0.74, help="length correction factor for latitude. Different by latitude.")

    local_metric = "LocalMetricConvDeep"
    global_metric = "GlobalMetricConvDeepL2"
    latent_dim = Proto(2, help="latent space for the global embedding. Not the local metric.")

    criteria = Proto("nn.MSELoss(reduction='mean')", help="evaluated to a criteria object")

    # Required data paths
    load_local_metric = Proto("models/local_metric.pkl", help="location to load the weight from")
    load_pairwise_ds = Proto(None, help="pre-computed distance between all samples, using the local metric")
    # load_top_k = Proto(None, help="pre-computed top-k, using the local metric")

    load_global_metric = Proto(None, help="location to load the pre-trained global metric."
                                          "example: `pretrain/global_metric.pkl`")

    top_k = Proto(None, help="k neighbors, 30 ~ 200 is usually a good range, depending on the size of the dataset.")
    neighbor_r = Proto(1, help="the threshold value for a sample being considered a `neighbor`.")
    term_r = Proto(1, help="the threshold value for a sample being considered a `neighbor`.")
    include_traj_neighbors = Proto(True, help="include neighbors in the trajectory in pairwise matrix."
                                              "This allows us to improve the accuracy without losing recall.")
    plan_steps = Proto(3, help="Search for neighbors that are 3 steps away from the current observed state.")

    sample_goal_r = Proto(1.2e-4 * 15, help="The radius within which the goals are sampled from. Default to 15 steps.")

    num_epochs = 2000
    batch_n = Proto(20, help="the batch size for the behavior sampler (as in VI)")
    H = Proto(20, help="planning horizon for the policy")

    gamma = Proto(1, help="discount factor")
    r_scale = Proto(1, help="scaling factor for the reward")

    eps_greedy = Proto(0.05, help="epsilon greedy factor. Explore when rn < eps")
    relabel_k = Proto(5, help="the k-interval for the relabeling")

    optim_epochs = 100
    optim_batch_size = 32

    target_update = Proto(0.9, help="when less than 1, does soft update. When larger or equal to 1, does hard update.")

    lr = 0.001
    weight_decay = 0

    k_fold = Proto(10, help="The k-fold validation for evaluating the planning module.")
    binary_reward = Proto(False, help="Uses a binary reward if true. Otherwise uses local metric instead.")
    checkpoint_interval = 50
    checkpoint_after = 0
    visualization_interval = 10


@cli_parse
class DEBUG:
    supervised_value_fn = Proto(False, help="True to turn on supervised grounding for the value function")
    # r_g_scale = Proto(1, help="Scaling the ground-truth distance for the supervised value fn baseline")
    oracle_planning = Proto(False, help="True to turn on oracle planner with ground-truth value function")
    oracle_eps_greedy = Proto(False, help="True to turn on eps-greedy with oracle distance metric.")
    real_r_distance = Proto(False, help="Overwritten when supervised_value_fn is True. This one toggles "
                                        "whether we use the real distance for the reward.")
    ground_truth_neighbor_r = Proto(False, help="when not zero, use this as a threshold for the ground"
                                                "truth neighbors")
    ground_truth_success = Proto(False, help="use the state-space distance as the termination condition.")

    # coordinate pretrain parameters
    pretrain_global = Proto(False, help="Pretrain the global embedding function")
    pretrain_lr = Proto(3e-7, help="anywhere between [1-3e-6 to -7] works well. 3e-7 is the best at epoch 200.")
    pretrain_num_epochs = Proto(800, help="Pretrain the global embedding function", dtype=int)
    pretrain_batch_size = 100
    pretrain_viz_interval = 1
    pretrain_logging_interval = 1

    # value_fn pretrain parameters (same as before)
    value_fn_pretrain_global = Proto(False, help="Pretrain the global embedding function")
    value_fn_pretrain_goal_r = Proto(None, help="the radius for the goal to be sampled")
    value_fn_pretrain_lr = Proto(3e-7,
                                 help="anywhere between [1-3e-6 to -7] works well. 3e-7 is the best at epoch 200.")
    value_fn_pretrain_num_epochs = Proto(800, help="Pretrain the global embedding function", dtype=int)
    value_fn_pretrain_batch_size = 100
    value_fn_pretrain_viz_interval = 1
    value_fn_pretrain_logging_interval = 1


# noinspection NonAsciiCharacters
def evaluate_metric(*, num_epochs, batch_n, ρ, ρ_goal, H, φ, Φ, N, oracle=None):
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
    :type ρ: function for sampling the state (and goals), the static distribution of states.
           ρ(s) := P(s).
    :type ρ_goal: function for sampling the goals. Allow selection of goals close to starting state.
    :param H: the planning horizon
    :param φ: the local metric
    :param Φ: the global metric we are trying to learn
    :param Φ: the target value function
    :param N: the neighbor search function
    :return:
    """
    from ml_logger import logger

    done, k = np.zeros(batch_n), np.zeros(batch_n)

    logger.split()
    for epoch in range(num_epochs + 1):

        with torch.no_grad():
            x, x_inds = ρ(batch_n)
            xg, xg_inds = ρ_goal(batch_n, x_inds if Args.sample_goal_r else None)

            logger.store_metrics(dt_epoch=logger.split())
            for plan_step in range(H):  # sampler rollout length

                ns, ds, ns_inds = N(x_inds)  # Size(batch, k)

                if Args.eps_greedy:
                    exploration_mask = np.random.random(Args.batch_n) < Args.eps_greedy

                    # cost = φ(x[:, None, ...], ns) + Φ(ns, xg[:, None, ...])
                    # note: now this is sliced
                    cost = Φ(ns, [g[:, None] for g in xg])

                    logger.store_metrics(metrics={"debug/explore_mask": exploration_mask})

                    _ = [torch.randint(0, len(c), size=tuple()).numpy()
                         if m else torch.argmin(c).cpu().numpy() for c, m in zip(cost, exploration_mask)]

                    if "plot correlation" and Args.visualization_interval and \
                            epoch % Args.visualization_interval == 0 and plan_step == 0:

                        import matplotlib.pyplot as plt
                        colors = ['#49b8ff', '#ff7575', '#66c56c', '#f4b247']
                        magic = [1, Args.lng_lat_correction]

                        fig = plt.figure(figsize=(4, 4))

                        min_x, max_x = float('inf'), - float('inf')
                        for i, (c, n, g) in enumerate(zip(cost, oracle(ns_inds), oracle(xg_inds))):
                            true_cost = np.linalg.norm((n - g) / magic, ord=Args.Lp, axis=-1)
                            plt.scatter(true_cost, c.cpu().numpy(), s=20, color=colors[i % len(colors)], alpha=0.3)
                            min_x = min(true_cost.min(), min_x)
                            max_x = max(true_cost.max(), min_x)

                        plt.title("Score vs Ground-truth Distance")
                        plt.xlabel("Ground Truth distance")
                        plt.ylabel("Score")
                        plt.xlim(min_x, max_x)
                        logger.savefig(f"debug/score_vs_gt_dist_{epoch:04d}.png")
                        del fig
                        plt.close()

                else:
                    cost = Φ(ns, [g[:, None] for g in xg])
                    _ = [torch.argmin(c).cpu().numpy() for c in cost]

                x_star = [n[ind] for ind, n in zip(_, ns)]
                x_star_inds = [n_inds[ind] for ind, n_inds in zip(_, ns_inds)]

                # DEBUG.ground_truth_success:
                magic = [1, Args.lng_lat_correction]
                _ = np.array([a - b for a, b in zip(oracle(x_star_inds), oracle(xg_inds))])
                success = np.linalg.norm(_ / magic, ord=Args.Lp, axis=-1) < Args.term_r

                done = success | (k >= H - 1)

                logger.store_metrics(ds=ds, episode_len=(k + 1)[np.ma.make_mask(done)])

                if done.sum():
                    logger.store_metrics(success=np.sum(success * done), done=done.sum())

                k = np.where(done, 0, k + 1)

                _x, _x_inds = ρ(batch_n)
                _xg, _xg_inds = ρ_goal(batch_n, _x_inds if Args.sample_goal_r else None)

                x = np.where(done[:, None, None, None], _x, x_star)
                x_inds = np.where(done, _x_inds, x_star_inds).astype(np.uint64)
                xg = np.where(done[:, None, None, None], _xg, xg)
                xg_inds = np.where(done, _xg_inds, xg_inds).astype(np.uint64)

        _ = logger.summary_cache.get_stats('success', 'done', default_stats="sum")
        logger.log_metrics_summary(ds="min_max", episode_len="quantile",
                                   success="sum", done="sum",
                                   key_stats={"debug/supervised_rg": "min_max"},
                                   key_values=dict(
                                       epoch=epoch, dt_epoch=logger.split(),
                                       success_rate=_['success/sum'] / _['done/sum'] if _[
                                           'done/sum'] else 0,
                                       timesteps=int(epoch * batch_n * H)))


# noinspection NonAsciiCharacters
def train(*, num_epochs, batch_n, optim_batch_size, ρ, ρ_goal, H, φ, Φ, N, Φ_target=None, oracle=None):
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
    :type ρ: function for sampling the state (and goals), the static distribution of states.
           ρ(s) := P(s).
    :type ρ_goal: function for sampling the goals. Allow selection of goals close to starting state.
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

    buffer = ReplayBuffer(2 * Args.H * Args.batch_n)

    optimizer = optim.Adam(Φ.parameters(), lr=Args.lr, weight_decay=Args.weight_decay)

    done, k = np.zeros(batch_n), np.zeros(batch_n)

    # evaluate the criteria
    criteria = eval(Args.criteria)

    import matplotlib.pyplot as plt

    logger.split()
    for epoch in range(num_epochs + 1):

        with torch.no_grad():
            # todo: visualize the value map here
            # todo: visualize the planned trajectories
            # todo: add evaluation here

            x, x_inds = ρ(batch_n)
            xg, xg_inds = ρ_goal(batch_n, x_inds if Args.sample_goal_r else None)

            logger.store_metrics(dt_epoch=logger.split())
            traj = defaultdict(list)
            for plan_step in range(H):  # sampler rollout length

                ns, ds, ns_inds = N(x_inds)  # Size(batch, k)

                if DEBUG.oracle_planning:
                    magic = [1, Args.lng_lat_correction]
                    cost = true_cost = [np.linalg.norm((n_ - g_[None, :]) / magic, ord=Args.Lp, axis=-1)
                                        for n_, g_ in zip(oracle(ns_inds), oracle(xg_inds))]
                    _ = [np.argmin(c) for c in cost]
                elif DEBUG.oracle_eps_greedy:
                    raise NotImplementedError
                    magic = [1, Args.lng_lat_correction]
                    cost = true_cost = [np.linalg.norm((n_ - g_[None, :]) / magic, ord=Args.Lp, axis=-1)
                                        for n_, g_ in zip(oracle(ns_inds), oracle(xg_inds))]

                    exploration_mask = (np.random.random() < Args.eps_greedy) \
                        if Args.eps_greedy else np.zeros(Args.batch_n)

                    _greedy = torch.argmin(cost.squeeze(dim=-1), dim=1).cpu().numpy()
                    _random = torch.randint(0, Args.top_k, size=(Args.batch_n,))
                    assert _random.shape == _greedy.shape, f"shape mismatch also kills kittens."
                    _ = np.where(exploration_mask, _random, _greedy)
                elif Args.eps_greedy:
                    exploration_mask = np.random.random(Args.batch_n) < Args.eps_greedy

                    # cost = φ(x[:, None, ...], ns) + Φ(ns, xg[:, None, ...])
                    # note: now this is sliced
                    cost = Φ(ns, [g[:, None] for g in xg])

                    logger.store_metrics(metrics={"debug/explore_mask": exploration_mask})

                    _ = [torch.randint(0, len(c), size=tuple()).numpy()
                         if m else torch.argmin(c).cpu().numpy() for c, m in zip(cost, exploration_mask)]

                    if "plot correlation" and epoch % Args.visualization_interval == 0 and plan_step == 0:
                        import matplotlib.pyplot as plt
                        colors = ['#49b8ff', '#ff7575', '#66c56c', '#f4b247']
                        magic = [1, Args.lng_lat_correction]

                        fig = plt.figure(figsize=(4, 4))

                        min_x, max_x = float('inf'), - float('inf')
                        for i, (c, n, g) in enumerate(zip(cost, oracle(ns_inds), oracle(xg_inds))):
                            true_cost = np.linalg.norm((n - g) / magic, ord=Args.Lp, axis=-1)
                            plt.scatter(true_cost, c.cpu().numpy(), s=20, color=colors[i % len(colors)], alpha=0.3)
                            min_x = min(true_cost.min(), min_x)
                            max_x = max(true_cost.max(), min_x)

                        plt.title("Score vs Ground-truth Distance")
                        plt.xlabel("Ground Truth distance")
                        plt.ylabel("Score")
                        plt.xlim(min_x, max_x)
                        logger.savefig(f"debug/score_vs_gt_dist_{epoch:04d}.png")
                        del fig
                        plt.close()

                else:
                    cost = Φ(ns, [g[:, None] for g in xg])
                    _ = [torch.argmin(c).cpu().numpy() for c in cost]

                x_star = [n[ind] for ind, n in zip(_, ns)]
                x_star_inds = [n_inds[ind] for ind, n_inds in zip(_, ns_inds)]

                # note: we can use cached distance instead, look up via sample index.
                if Args.binary_reward:
                    if DEBUG.ground_truth_success:
                        magic = [1, Args.lng_lat_correction]
                        _ = np.array([a - b for a, b in zip(oracle(x_star_inds), oracle(xg_inds))])
                        success = np.linalg.norm(_ / magic, ord=Args.Lp, axis=-1) < Args.term_r
                    else:
                        # success = np.stack([si == gi for si, gi in zip(x_star_inds, xg_inds)])
                        success = np.stack([gi in n_inds for n_inds, gi in zip(ns_inds, xg_inds)])
                    r = 1 - success  # only used for reporting.
                elif DEBUG.real_r_distance:
                    magic = [1, Args.lng_lat_correction]
                    r = np.array([(a - b) / magic for a, b in zip(oracle(x_star_inds), oracle(xg_inds))])
                    success = r < 2e-4
                else:
                    r = φ(x_star, xg).squeeze(-1)
                    if DEBUG.ground_truth_success:
                        magic = [1, Args.lng_lat_correction]
                        _ = np.array([a - b for a, b in zip(oracle(x_star_inds), oracle(xg_inds))])
                        success = np.linalg.norm(_ / magic, ord=Args.Lp, axis=-1) < Args.term_r
                    else:
                        success = (r <= Args.term_r).cpu().numpy()
                    r = r.cpu().numpy()

                done = success | (k >= H - 1)

                # note: for HER, preserving the rollout structure improves relabel efficiency.
                traj["x"].append(x)
                traj["next"].append(x_star)
                traj["r"].append(r)
                traj["goal"].append(xg)
                traj["done"].append(done)
                traj['success'].append(success)

                # for debug with supervised training
                traj['s'].append(oracle(x_inds))
                traj['s_next'].append(oracle(x_star_inds))
                traj['s_goal'].append(oracle(xg_inds))

                logger.store_metrics(ds=ds, r=r, episode_len=(k + 1)[np.ma.make_mask(done)])
                if done.sum():
                    logger.store_metrics(success=np.sum(success * done), done=done.sum())

                k = np.where(done, 0, k + 1)

                _x, _x_inds = ρ(batch_n)
                _xg, _xg_inds = ρ_goal(batch_n, _x_inds if Args.sample_goal_r else None)

                x = np.where(done[:, None, None, None], _x, x_star)
                x_inds = np.where(done, _x_inds, x_star_inds).astype(np.uint64)
                xg = np.where(done[:, None, None, None], _xg, xg)
                xg_inds = np.where(done, _xg_inds, xg_inds).astype(np.uint64)

            traj = {k: np.array(l) for k, l in traj.items()}

            # avg_success = sum(logger.summary_cache.get('success', 0)) / sum(logger.summary_cache.get('done', 1))
            # if epoch < 20:  # only log video when the success rate improves.
            #     max_success = 0.  # let it roll down.
            #     last_checkpoint = epoch
            # elif avg_success > max_success:
            #     logger.log_data(dict(s=traj['s'], s_goal=traj['s_goal']), f"plans/traj_{epoch:04d}.pkl", overwrite=True)
            #     if Args.checkpoint_interval and last_checkpoint < (epoch - Args.checkpoint_interval):
            #         last_checkpoint = epoch
            #         logger.save_module(Φ, f'models/global_metric_{epoch:05d}.pkl', show_progress=True, chunk=100_000_000)
            #
            # max_success = max_success * 0.9 + avg_success * 0.1

            if epoch > Args.checkpoint_after and \
                    Args.checkpoint_interval and \
                    epoch % Args.checkpoint_interval == 0:
                logger.save_module(Φ, f'models/global_metric_{epoch:05d}.pkl', show_progress=True, chunk=100_000_000)

            # done: need to preserve rollout structure
            buffer.extend(**{k: v.reshape(-1, *v.shape[2:]) for k, v in traj.items() if k != "done"})
            logger.store_metrics(metrics={"traj/r": traj['r']})
            # done: Hindsight Experience Relabel
            if Args.relabel_k:
                for goals, state_goals in zip(traj['next'][::Args.relabel_k], traj['s_next'][::Args.relabel_k]):
                    new_goals = np.tile(goals, [H, 1, 1, 1, 1])
                    new_state_goals = np.tile(state_goals, [H, 1, 1])
                    if Args.binary_reward:
                        if DEBUG.ground_truth_success:
                            magic = [1, Args.lng_lat_correction]
                            _ = traj['s_next'] - new_state_goals
                            success = np.linalg.norm(_ / magic, ord=Args.Lp, axis=-1) < Args.term_r
                        else:
                            # note: identity are fine for relabeling.
                            success = np.logical_and(*np.equal(traj['s_next'], new_state_goals).T)
                        new_r = 1 - success  # only used for reporting.
                    elif DEBUG.real_r_distance:
                        magic = [1, Args.lng_lat_correction]
                        _ = traj['s_next'] - new_state_goals
                        new_r = np.linalg.norm(_ / magic, ord=Args.Lp, axis=-1)
                        success = new_r < 2e-4
                    else:
                        new_r = φ(traj['next'], new_goals).squeeze(-1)
                        success = torch.stack([_ <= Args.term_r for _ in new_r]).cpu().numpy()
                        new_r = new_r.cpu().numpy()

                    buffer.extend(**{
                        k: v.reshape(-1, *v.shape[2:])
                        for k, v in dict(
                            x=traj['x'],
                            next=traj['next'],
                            r=new_r,
                            goal=new_goals,
                            success=success,
                            # for supervised training
                            s=traj['s'],
                            s_next=traj['s_next'],
                            s_goal=new_state_goals,
                        ).items()
                    })

        for i in range(Args.optim_epochs):
            paths = buffer.sample(optim_batch_size)
            r = torch.tensor(paths['r'], dtype=torch.float, device=Args.device)
            success = torch.tensor(paths['success'].astype('float'), dtype=torch.float, device=Args.device)

            # done: add target value function.
            values = torch.stack(Φ(paths['x'], paths['goal'])).squeeze()

            with torch.no_grad():
                if DEBUG.supervised_value_fn:
                    # we want to supervise the value function like this
                    magic = [1, Args.lng_lat_correction]
                    # r_g = np.linalg.norm((paths['s'] - paths['s_goal']) / magic, ord=Args.Lp, axis=-1) * DEBUG.r_g_scale
                    r_g = np.linalg.norm((paths['s'] - paths['s_goal']) / magic
                                         * DEBUG.oracle_scaling, ord=Args.Lp, axis=-1)
                    target_values = torch.tensor(r_g, dtype=torch.float32, device=Args.device)
                    logger.store_metrics(metrics={"debug/supervised_rg": r_g})
                elif DEBUG.real_r_distance:
                    mask = 1. - success
                    target_values = r + Args.gamma * mask * Φ_target(paths['next'], paths['goal']).squeeze()
                # done: use success (saved in traj) to label termination
                elif Args.binary_reward:
                    target_values = r * Args.r_scale \
                                    + Args.gamma * r * Φ_target(paths['next'], paths['goal']).squeeze()
                else:
                    mask = 1. - success
                    target_values = r * Args.r_scale \
                                    + Args.gamma * mask * Φ_target(paths['next'], paths['goal']).squeeze()

            true_delta = torch.abs(values - target_values).mean()
            assert values.shape == target_values.shape, \
                f"broadcasting kills kittens. {values.shape}, {target_values.shape}"

            δ = criteria(values, target_values).squeeze()
            logger.store_metrics(loss=δ.item(),
                                 true_delta=true_delta.item(),
                                 value=values.mean().item(),
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
            elif (epoch * Args.optim_steps + _) % Args.target_update == 0:  # <-- hard target update
                Φ_target.load_state_dict(Φ.state_dict())

        _ = logger.summary_cache.get_stats('success', 'done', default_stats="sum")
        logger.log_metrics_summary(
            ds="min_max", r="quantile", episode_len="quantile",
            success="sum", done="sum",
            key_stats={"debug/supervised_rg": "min_max"},
            key_values=dict(epoch=epoch, dt_epoch=logger.split(),
                            success_rate=_['success/sum'] / _['done/sum'] if _['done/sum'] else 0,
                            timesteps=int(epoch * batch_n * H)))

        if Args.visualization_interval and epoch % Args.visualization_interval == 0:
            if Args.latent_dim in [2, 3]:
                logger.split("visualize_embedding")
                eval(f"visualize_embedding_{Args.latent_dim}d_image")(  # LOL I know. I'm a terrible person.
                    Args.env_id, Φ.embed, f"figures/embedding/embed_{Args.latent_dim}d_{epoch:06}.png")

            logger.store_metrics(metrics={"dt_vis": logger.split("visualize_embedding")})

        # todo: clear buffer to learn with only on-policy samples
        # todo: keep filling the buffer, and learn from recent experiences.
        # buffer.clear()


def pairwise_fn(φ, xs, xs_slice, chunk=1, row_chunk=8000):
    from tqdm import tqdm

    return torch.cat([
        torch.cat([
            φ(*torch.broadcast_tensors(_xs.unsqueeze(0), _xs_slice.unsqueeze(1))).squeeze(-1)
            for _xs_slice in tqdm(tslice(xs_slice, chunk=chunk), desc="computing pair-wise distance")
        ]) for _xs in tslice(xs, chunk=row_chunk)], dim=-1)


def ground_truth_neighbor_factory(obs, xs, r, lat_correction, soft=False):
    """Factory for ground-truth neighbors within distance r.

    :param obs: observation samples
    :param xs: state-space ground-truth
    :param r: radius for the neighborhood
    :param lat_correction:
    :param soft: Not used
    :return: neighbors, distances, neighbor indices (in the dataset)
    """

    magic = [1., lat_correction]
    with torch.no_grad():
        _ = torch.tensor(xs / magic, dtype=torch.float32)
        pairwise_ds = torch.norm(_[:, None, :] - _[None, :, :], p=2, dim=-1)
        data_size = len(pairwise_ds)
        # 1. set diag to inf
        pairwise_ds[torch.eye(data_size, dtype=torch.uint8)] = float("inf")
        # 2. get indices of filtered
        full_range = torch.arange(data_size)
        top_inds = [full_range[row <= r] for row in pairwise_ds]
        top_ds = [ds[inds] for ds, inds in zip(pairwise_ds, top_inds)]

    del pairwise_ds

    # too big, do not log.
    # from ml_logger import logger
    # logger.log_image(pairwise_ds, f"figures/debug/ground_truth_pairwise_ds.png")

    def N(x_indices):
        nonlocal xs, top_ds, top_inds
        # around a pytorch bug: https://github.com/pytorch/pytorch/issues/20697
        return zip(*[(obs[top_inds[_].tolist()], top_ds[_], top_inds[_]) for _ in x_indices])

    return N


def neighbor_factory(xs, trajs, φ, top_k=None, step_threshold=None, include_traj_neighbors=True):
    """Factory for get_neighbor functions.

    This version differs from the state-space neighbor factor in that this one
    computes the distance matrix `ds` piece-by-piece. This is because the raw
    images for computing the entire matrix is 740 GB, too large to fit on a single
    card with CUDA.

    **Important**: neighbors include actual neighbors in the sample trajectories. We
    need to local_metric function to return with high-precision, which reduces recall.

    To improves the recall, we use the actual neighbors. How much this is relied upon
    depends on the problem (the domain) and the network architecture. In a real dataset
    we always have this information, and we ought to use it.

    :param xs: Size(batch_n, feat_n) the complete set of samples from which we want
      to build the graph.
    :type trajs: Size(n, 1) the label for which trajectory these sample come from
    :param φ: the local metric function we use to compute the 1-step neighbors.
            Given Torch.Tensor, do NOT wrap with `torchify`.
    :param top_k: Int(top k inputs) k for top-k neighbors
    :param step_threshold: The threshold for each planning step
    :param include_traj_neighbors: includes ground-truth neighbors from each trajectory.
    :return: a neighbor function.
    """
    import torch
    from tqdm import tqdm
    from ml_logger import logger

    with torch.no_grad():
        if Args.load_pairwise_ds:
            with logger.PrefixContext(Args.load_pairwise_ds):
                chunks = sorted(logger.glob("data/chunk*.pkl"))
                _ds = np.concatenate([logger.load_pkl(p)[0] for p in tqdm(chunks, desc="load pairwise matrix")])
            with logger.SyncContext():
                logger.log_text(Args.load_pairwise_ds + '\n', filename="data/chunk-list.txt")
                logger.log_text("\n".join(chunks), filename="data/chunk-list.txt")

            ds = torch.tensor(_ds, device=Args.device, dtype=torch.float32)

        else:
            _xs = torch.tensor(xs, device=Args.device, dtype=torch.float32)
            ds = pairwise_fn(φ=φ, xs=_xs, xs_slice=_xs, chunk=1)

        ds[torch.eye(len(ds), device=Args.device, dtype=torch.uint8)] = float('inf')

        logger.log_image(ds.cpu().numpy()[:100, :100], f"figures/debug/ds_pairwise.png", cmap='viridis_r',
                         normalize=[0, 2.5])
        logger.log_image((ds * ds <= step_threshold).cpu().numpy().astype(float)[:100, :100],
                         f"figures/debug/ds_pairwise_not.png", normalize=[0, 1])

        if include_traj_neighbors:
            # use int type to allow negative numbers
            for traj in trajs:
                for i, j in zip(traj[:-1], traj[1:]):
                    ds[i, j] = step_threshold
                    ds[j, i] = step_threshold

            logger.log_image(ds.cpu().numpy()[:100, :100],
                             f"figures/debug/ds_pairwise_with_in_trajectory_neighbors.png",
                             cmap='viridis', normalize=[None, 2.5])

        if top_k:
            top_ds, top_inds = [_.cpu().numpy() for _ in torch.topk(ds, k=top_k, dim=1, largest=False, sorted=True)]
        else:
            # add threshold connection instead.
            full_range = torch.arange(len(ds), device=Args.device)
            top_inds = [full_range[row <= step_threshold].cpu().numpy() for row in ds]
            top_ds = [row[_].cpu().numpy() for row, _ in zip(ds, top_inds)]

    # check for samples with no neighbors
    for i, inds in enumerate(top_inds):
        if len(inds) == 0:
            raise RuntimeError("Point has no neighbor! Is your term_r too stringent? Check the visualization under"
                               "figures/<env-name>_score_vs_gt.png")

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
        ns_inds = [_.difference({ind}) for _, ind in zip(sofar, x_inds)]

        # watch out for a pytorch bug: https://github.com/pytorch/pytorch/issues/20697
        return zip(*[(xs[_], ds[x_ind].cpu().numpy()[_], _) for x_ind, _ in zip(x_inds, map(list, ns_inds))])

    return N_multistep


def main(_DEBUG=None, map_reduce=None, evaluate=False, **kwargs, ):
    from ml_logger import logger

    Args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Args.update(kwargs)
    if _DEBUG is not None:
        DEBUG.update(_DEBUG)

    if map_reduce:
        logger.log_params(map_reduce=map_reduce)
    else:
        logger.log_params(Args=vars(Args), DEBUG=vars(DEBUG))

    GlobalMetric = globals().get(Args.global_metric)
    logger.log_text('\n'.join([k for k in globals().keys() if "GlobalMetric" in k]), "debug/available_global_models.md")

    np.random.seed(Args.seed)
    torch.manual_seed(Args.seed)
    # see https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if True:  # load local metric
        cprint('loading local metric', "yellow", end="... ")

        # hard code args for experiment
        f_local_metric = globals()[Args.local_metric](1, Args.latent_dim)
        logger.log_text(str(f_local_metric), filename="models/f_local_metric.txt")
        logger.load_module(f_local_metric, Args.load_local_metric)
        # note: DataParallel is issue with batch size 1 in binary function
        # if torch.cuda.device_count() > 1:
        #     logger.print(f"f_local_metric using {torch.cuda.device_count()} GPUs", color="yellow")
        #     f_local_metric = nn.DataParallel(f_local_metric)
        f_local_metric.to(Args.device)
        f_local_metric.eval()
        cprint('✔done', 'green')

    if True:  # get rope dataset
        cprint('loading environment dataset', "yellow", end="... ")

        # collect sample here
        from streetlearn import StreetLearnDataset
        from os.path import expanduser

        streetlearn = StreetLearnDataset(expanduser(Args.data_path), Args.street_view_size, Args.street_view_mode)
        streetlearn.select_all()
        Args.streetlearn_bbox = streetlearn.bbox

        all_images = streetlearn.images[:, None, ...].astype(np.float32) / 255
        all_states = streetlearn.lng_lat
        all_trajs = streetlearn.trajs
        all_range = torch.arange(len(all_states))

        # 2019-07-30 found bug, neighbors might not be found properly
        traj_labels, step_labels = [], []
        for traj_label, traj in enumerate(streetlearn.trajs):
            l = len(traj)
            traj_labels.append(np.ones(l) * traj_label)
            step_labels.append(np.arange(l))

        traj_labels, step_labels = np.concatenate(traj_labels), np.concatenate(step_labels)

        # Need a streetview version
        # with torch.no_grad():
        #     from plan2vec.plotting.maze_world.visualize_image_maze import connect_the_dots, Args as PlotArgs
        #     PlotArgs.update(vars(Args))
        #     connect_the_dots(trajs=raw_lng_lats,
        #                      img_trajs=raw_images,
        #                      f_local_metric=f_local_metric)

        cprint('✔done', 'green')

    if map_reduce:
        raise NotImplementedError('Not tested')
        with torch.no_grad():
            import math
            k, n = map_reduce['k'], map_reduce['n']
            _ = torch.tensor(all_images, device=Args.device, dtype=torch.float32)
            chunk = math.ceil(len(_) / n)
            ds_slice = pairwise_fn(φ=f_local_metric, xs=_, xs_slice=_[chunk * k: chunk * k + chunk], chunk=1)
            with logger.SyncContext():  # make sure we finish upload before exit
                logger.log_data(ds_slice.cpu().numpy(), f"data/chunk_{k:02d}_{n:02d}.pkl", overwrite=True)
                logger.log_line(f"uploaded pairwise chunk {k:02d}/{n}", color="green", file="data/upload.log")
                if k == 0:
                    logger.log_data(all_states, f"data/all_states.pkl", overwrite=True)
            exit()

    # allow the power config ONLY for ResNet18L2
    ext = {"p": Args.Lp} if Args.global_metric == "ResNet18L2" else {}

    global_metric = GlobalMetric(1, latent_dim=Args.latent_dim, **ext).to(Args.device)
    if not evaluate:
        target_global_metric = GlobalMetric(1, latent_dim=Args.latent_dim, **ext).to(Args.device)
        target_global_metric.load_state_dict(global_metric.state_dict())

    logger.log_text(str(global_metric), filename="models/global_metric.txt")
    # note: DataParallel has issue with batch size == 1, in binary function
    # if torch.cuda.device_count() > 1:
    #     logger.print(f"global_metric using {torch.cuda.device_count()} GPUs", color="yellow")
    #     global_metric = nn.DataParallel(global_metric)
    #     global_metric.embed = global_metric.module.embed
    #     target_global_metric = nn.DataParallel(target_global_metric)

    # always cache the images for the visualization
    if Args.visualization_interval:
        cache_images(all_images, all_states, Args.lng_lat_correction)

    coord_scaling = 1 / (all_states.max(0) - all_states.min(0))
    logger.print('DEBUG: locally writing DEBUG.oracle_scaling', color="red")
    logger.print('DEBUG: locally writing DEBUG.oracle_mean', color="red")
    DEBUG.oracle_scaling = coord_scaling
    DEBUG.oracle_mean = all_states.mean(0)
    # log the coord scaling.
    logger.log_params(Args=dict(coord_scaling=coord_scaling.tolist()),
                      DEBUG=dict(oracle_scaling=coord_scaling,
                                 oracle_mean=DEBUG.oracle_mean))

    if Args.load_global_metric:
        logger.log(f"Loading global Metric from checkpoint {Args.load_global_metric}", color="green")
        if Args.load_global_metric_matcher:
            logger.load_module(global_metric, Args.load_global_metric, matcher=Args.load_global_metric_matcher)
        else:
            logger.load_module(global_metric, Args.load_global_metric)

        if not evaluate and Args.latent_dim in [2, 3]:
            eval(f"visualize_embedding_{Args.latent_dim}d_image")(  # LOL I know.
                Args.env_id, global_metric.embed,
                f"loaded_global_metric/embed_{Args.latent_dim}d.png")

        if "plot correlation":
            with torch.no_grad():
                _ = torch.tensor(all_images[:1000], device=Args.device)
                _r = torch.tensor(list(all_images)[999::-1], device=Args.device)
                all_cost = global_metric(_, _r)
                magic = [1, Args.lng_lat_correction]
                all_true_cost = np.linalg.norm((all_states[:1000] - all_states[999::-1]) / magic, ord=Args.Lp, axis=-1)

            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(4, 4))
            plt.scatter(all_true_cost, all_cost.cpu().numpy(), s=20, color="#23aaff", alpha=0.3)
            plt.title("Score vs Ground-truth Distance")
            plt.xlabel("Ground Truth distance")
            plt.ylabel("Score")
            plt.xlim(all_true_cost.min(), all_true_cost.max())
            logger.savefig("debug/Score vs Ground-truth Distance.png")
            del fig
            plt.close()

    # first run the pretrain
    if DEBUG.pretrain_global:
        criteria = nn.MSELoss(reduce='sum')
        logger.log_params(Args=dict(pretrain_reduce="sum"))
        optimizer = optim.Adam(global_metric.parameters(), lr=DEBUG.pretrain_lr, weight_decay=Args.weight_decay)
        # let's normalize the labels to make it easier to learn.
        # todo: normalize w.r.t step size
        y = torch.tensor((all_states - all_states.mean(0)) * coord_scaling,
                         device=Args.device, dtype=torch.float32)
        x = torch.tensor(all_images, device=Args.device, dtype=torch.float32)

        datasize = len(all_states)
        mask_ratio = DEBUG.pretrain_batch_size / datasize

        import math
        from tqdm import trange
        logger.split()

        for pre_epoch in trange(DEBUG.pretrain_num_epochs + 1, desc="pretrain the embedding"):
            for i in range(math.ceil(1 / mask_ratio)):
                mask = torch.rand(datasize, device=Args.device) < mask_ratio
                y_hat = global_metric.embed(x[mask])
                loss = criteria(y_hat, y[mask])
                loss.backward()
                optimizer.step()
                logger.store_metrics(metrics={
                    "pretrain/loss": loss.detach().cpu().item(),
                    "pretrain/dt_step": logger.split()
                })

            if DEBUG.pretrain_viz_interval and pre_epoch % DEBUG.pretrain_viz_interval == 0:
                if Args.latent_dim in [2, 3]:
                    logger.split("visualize_embedding")
                    eval(f"visualize_embedding_{Args.latent_dim}d_image")(  # LOL I know. I'm a terrible person.
                        Args.env_id, global_metric.embed,
                        f"pretrain/embedding/embed_{Args.latent_dim}d_{int(pre_epoch):06}.png")

                logger.store_metrics(metrics={"pretrain/dt_vis": logger.split("visualize_embedding")})

            if not DEBUG.pretrain_logging_interval or pre_epoch % DEBUG.pretrain_logging_interval == 0:
                logger.log_metrics_summary(default_stats='quantile', key_values={
                    "pretrain/epoch": pre_epoch
                })

        # now save the pretrained-model.
        logger.save_module(global_metric, "pretrain/global_metric_fn.pkl", show_progress=True, chunk=100_000_000)

    def sample_obs(batch_n):
        """Sample a batch of `batch_n` observations and indices
         from the the dataset, with *replacement*.

        :param batch_n: (int) the size for the returned batch
        :return: Size(batch_n, 1, 64, 64), Size(batch_n)
        """
        inds = np.random.randint(0, len(all_images), size=batch_n)
        return all_images[inds], inds

    def oracle(indices):
        """
        The oracle for states, from a batch of indices.

        :param indices: Size(batch_n)
        :return:
        """
        return [all_states[inds] for inds in indices]

    if Args.sample_goal_r:
        N_goal = ground_truth_neighbor_factory(obs=all_images, xs=all_states, r=Args.sample_goal_r,
                                               lat_correction=Args.lng_lat_correction)

        visualize_neighbors_r(all_states, N_goal, oracle,
                              Args.lng_lat_correction, r=Args.sample_goal_r,
                              path=f"debug/goals_{Args.sample_goal_r:.0E}.png")
        visualize_neighbors_r(all_states, N_goal, oracle,
                              Args.lng_lat_correction, r=Args.sample_goal_r,
                              path=f"debug/goals_{Args.sample_goal_r:.0E}.pdf")

    def choice(a, b):
        assert len(a) == len(b), f"a and b need to have the same length."
        selection = np.random.randint(0, len(a))
        return a[selection], b[selection]

    def sample_goals(batch_n, x_inds=None):
        """Sample a batch of `batch_n` observations and indices
         from the the dataset, with *replacement*.

        :param batch_n: (int) the size for the returned batch
        :param x_inds: An optional list of indices for the start position. To limit goals to reachable area
        :return: Size(batch_n, 1, 64, 64), Size(batch_n)
        """
        if Args.sample_goal_r:
            assert x_inds is not None
            ns_obs, ns_ds, ns_inds = N_goal(x_inds)
            images, inds = zip(*[choice(*_) for _ in zip(ns_obs, ns_inds)])
            # todo: double check if this works under GPU run.
            return images, [i.cpu().item() for i in inds]
        else:
            inds = np.random.randint(0, len(all_images), size=batch_n)
            return all_images[inds], inds

    if DEBUG.ground_truth_neighbor_r:
        assert Args.plan_steps == 1, "the DEBUG.ground_truth_neighbor_r overrides the planning step option."
        N = ground_truth_neighbor_factory(obs=all_images, xs=all_states, r=DEBUG.ground_truth_neighbor_r,
                                          lat_correction=Args.lng_lat_correction)
        visualize_neighbors_r(all_states, N, oracle,
                              Args.lng_lat_correction, r=DEBUG.ground_truth_neighbor_r,
                              path=f"debug/neighbors_{DEBUG.ground_truth_neighbor_r:.0E}_r.png")
        visualize_neighbors_r(all_states, N, oracle,
                              Args.lng_lat_correction, r=DEBUG.ground_truth_neighbor_r,
                              path=f"debug/neighbors_{DEBUG.ground_truth_neighbor_r:.0E}_r.pdf")
    else:
        n_fn = neighbor_factory(xs=all_images, trajs=all_trajs, φ=f_local_metric,
                                top_k=Args.top_k, step_threshold=Args.neighbor_r,
                                include_traj_neighbors=Args.include_traj_neighbors)
        N = lambda x_inds: n_fn(x_inds, Args.plan_steps)
        visualize_neighbors_r(all_states, N, oracle,
                              Args.lng_lat_correction, path=f"debug/neighbors_{Args.plan_steps:02d}_steps.png")
        visualize_neighbors_r(all_states, N, oracle,
                              Args.lng_lat_correction, path=f"debug/neighbors_{Args.plan_steps:02d}_steps.pdf")

    # xs, xs_inds = sample_obs(10)
    # ns, ds, ns_inds = N(xs_inds)
    # visualize_neighbors(xs, ns, xs_inds, prefix="figures/neighbors")
    # visualize_neighbor_states(all_states, xs_inds, ns_inds, f"figures/neighbors/neighbor_states.png")

    if DEBUG.value_fn_pretrain_goal_r:
        # this part just visualizes the neighbor function, nothing special
        pretrain_pair_N = ground_truth_neighbor_factory(obs=all_images, xs=all_states, r=DEBUG.value_fn_pretrain_goal_r,
                                                        lat_correction=Args.lng_lat_correction)

        visualize_neighbors_r(all_states, pretrain_pair_N, oracle,
                              Args.lng_lat_correction, r=DEBUG.value_fn_pretrain_goal_r,
                              path=f"value_fn_pretrain/figures/neighbors_{Args.sample_goal_r:.0E}.png")
        visualize_neighbors_r(all_states, pretrain_pair_N, oracle,
                              Args.lng_lat_correction, r=DEBUG.value_fn_pretrain_goal_r,
                              path=f"value_fn_pretrain/figures/neighbors_{Args.sample_goal_r:.0E}.pdf")

    if DEBUG.value_fn_pretrain_global:
        criteria = nn.MSELoss(reduce='sum')
        logger.log_params(Args=dict(value_fn_pretrain_reduce="sum"))
        optimizer = optim.Adam(global_metric.parameters(), lr=DEBUG.value_fn_pretrain_lr,
                               weight_decay=Args.weight_decay)
        # let's normalize the labels to make it easier to learn.
        # todo: normalize w.r.t step size
        y = torch.tensor((all_states - all_states.mean(0)) * coord_scaling,
                         device=Args.device, dtype=torch.float32)
        x = torch.tensor(all_images, device=Args.device, dtype=torch.float32)

        datasize = len(all_states)
        mask_ratio = DEBUG.value_fn_pretrain_batch_size / datasize

        import math
        from tqdm import trange
        logger.split()
        for pre_epoch in trange(DEBUG.value_fn_pretrain_num_epochs + 1, desc="value_fn_pretrain the embedding"):
            for i in range(math.ceil(1 / mask_ratio)):
                # todo: move this outside
                mask = torch.rand(datasize, device=Args.device) < mask_ratio
                if DEBUG.value_fn_pretrain_goal_r:
                    x_inds = all_range[mask]
                    ns_obs, ns_ds, ns_inds = pretrain_pair_N(x_inds)
                    mask_next = [torch.randint(0, len(inds), ()) for inds in ns_inds]
                    if i == 0 and DEBUG.value_fn_pretrain_viz_interval and \
                            pre_epoch % DEBUG.value_fn_pretrain_viz_interval == 0:
                        from plan2vec.plotting.streetlearn.neighbors import visualize_neighbors
                        visualize_neighbors(all_states, x_inds[-2:], ns_inds[-2:], oracle,
                                            Args.lng_lat_correction, DEBUG.value_fn_pretrain_goal_r,
                                            title="Value Fn Pretrain Pair Range",
                                            path=f'value_fn_pretrain/figures/pair_range_{pre_epoch:06d}.png')
                        visualize_neighbors(all_states, x_inds[-2:], ns_inds[-2:], oracle,
                                            Args.lng_lat_correction, DEBUG.value_fn_pretrain_goal_r,
                                            title="Value Fn Pretrain Pair Range",
                                            path=f'value_fn_pretrain/figures/pair_range_{pre_epoch:06d}.pdf')
                else:
                    mask_next = mask[torch.randperm(mask.shape[0])]
                d_hat = global_metric(x[mask], x[mask_next])

                if DEBUG.value_fn_pretrain_goal_steps:
                    print('goal_steps is not implemented')

                d = torch.norm(y[mask] - y[mask_next], p=2, dim=-1, keepdim=True)

                assert d_hat.shape == d.shape, "y and y_hat need to have identical shape"
                assert len(d_hat.shape) == 2, "the d_hat should be 2-dimensional"
                loss = criteria(d_hat, d)
                loss.backward()
                optimizer.step()
                logger.store_metrics(metrics={
                    "value_fn_pretrain/loss": loss.detach().cpu().item(),
                    "value_fn_pretrain/dt_step": logger.split()
                })

            if DEBUG.value_fn_pretrain_viz_interval and pre_epoch % DEBUG.value_fn_pretrain_viz_interval == 0:
                if Args.latent_dim in [2, 3]:
                    logger.split("visualize_embedding")
                    eval(f"visualize_embedding_{Args.latent_dim}d_image")(  # LOL I know. I'm a terrible person.
                        Args.env_id, global_metric.embed,
                        f"value_fn_pretrain/embedding/embed_{Args.latent_dim}d_{int(pre_epoch):06}.png")

                logger.store_metrics(metrics={"value_fn_pretrain/dt_vis": logger.split("visualize_embedding")})

            if not DEBUG.value_fn_pretrain_logging_interval or pre_epoch % DEBUG.value_fn_pretrain_logging_interval == 0:
                logger.log_metrics_summary(default_stats='quantile', key_values={
                    "value_fn_pretrain/epoch": pre_epoch
                })

        # now save the value_fn_pretrained-model.
        logger.save_module(global_metric, "value_fn_pretrain/global_metric_fn.pkl", show_progress=True,
                           chunk=10_000_000)

    if evaluate:
        evaluate_metric(num_epochs=Args.num_epochs, batch_n=Args.batch_n,
                        ρ=sample_obs,
                        ρ_goal=sample_goals,
                        H=Args.H,
                        φ=torchify(f_local_metric, dtype=torch.float32, input_only=True, device=Args.device),
                        Φ=sliced_helper(
                            torchify(global_metric, dtype=torch.float32, input_only=True, device=Args.device)),
                        N=N,
                        oracle=oracle)
    else:
        train(num_epochs=Args.num_epochs, batch_n=Args.batch_n,
              optim_batch_size=Args.optim_batch_size,
              ρ=sample_obs,
              ρ_goal=sample_goals,
              H=Args.H,
              φ=torchify(f_local_metric, dtype=torch.float32, input_only=True, device=Args.device),
              Φ=sliced_helper(torchify(global_metric, dtype=torch.float32, input_only=True, device=Args.device)),
              N=N,
              Φ_target=torchify(target_global_metric, input_only=True,
                                device=Args.device) if Args.target_update else None,
              oracle=oracle)


if __name__ == "__main__":
    from ml_logger import logger
    from plan2vec_experiments import instr

    _ = instr(main, seed=5 * 100)

    if True:
        Args.env_id = "CMazeDiscreteImgIdLess-v0"
        Args.n_rollouts = 20
        Args.batch_n = 10
        local_metric_exp_path = "episodeyang/plan2vec/2019/05-07/c-maze-image/c_maze_local_metric/21.19/10.332015"
        Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"
        _()

    elif "map-reduce":
        cprint('Computing pairwise with map-reduce', 'green')
        import jaynes

        jaynes.config("learnfair-gpu")

        jaynes.run(_, map_reduce=dict(k=k, n=40))
        jaynes.listen()

    elif False:
        cprint('Training on cluster', 'green')
        import jaynes

        jaynes.config("learnfair-gpu")
        jaynes.run(_)
        jaynes.listen()
