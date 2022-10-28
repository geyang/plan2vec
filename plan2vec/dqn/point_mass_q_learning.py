import numpy as np

import pandas
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from termcolor import cprint
from ml_logger import logger

from ge_world import IS_PATCHED
from params_proto.neo_proto import ParamsProto, Proto

assert IS_PATCHED is True, "need patch"

# from plan2vec.plotting.embedding_2d import visualize_embedding_2d
from plan2vec.plotting.plot_q_value import visualize_q_2d
from plan2vec.mdp.helpers import make_env, sample_gen_fn, render_gen_fn
from plan2vec.mdp.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from plan2vec.mdp.wrappers.subproc_vec_env import SubprocVecEnv
from plan2vec.mdp.models import QMlp, QConv, QSharedEmbedding, QSharedEmbeddingConv, QL2Embed, QL2EmbedModel, \
    QL2EmbedConv, QL2EmbedModelConv, QL2EmbeddingPassive, QL2EmbeddingPassiveConv, QDeepConv, \
    ResNet18L2Q, ResNet18CoordL2Q, ResNet18CoordQ

from plan2vec.schedules import Schedule, LinearAnneal
from plan2vec.mdp.imaginator import imaginator
from torch_utils import torchify


class Args(ParamsProto):
    env_id = Proto('GoalMassDiscreteIdLess-v0')
    obs_key = "x"
    goal_key = "goal"
    start_seed = Proto(0, dtype=int, help='the random seed for the experiment')
    num_envs = Proto(10, dtype=int, help='the number of parallel samplers')

    batch_timesteps = 50
    num_episodes = 2000
    gamma = 0.98
    optim_steps = Proto(10, dtype=int, help="number of optimization steps to do after sampling")
    optim_batch_size = Proto(64, dtype=int, help='batch_size for the optimization')

    learning_mode = Proto("mdp", help="one of [`mdp`, `passive`]. ")
    n_rollouts = Proto(None, help="if not None, becomes the sample limit for the training generator")

    # HER configs
    her_strategy = Proto("episode", help="OneOf[`episode`, `future`, `random`] from the HER paper. "
                                         "Only the `episode` mode is implemented")
    her_k_goals = Proto(10, dtype=int, help="number of goals to re-sample per batch.")

    # losses
    forward_coef = Proto(None, dtype=float,
                         help="loss coefficient for forward model loss. Default to None b/c only used with T q_fn.")
    sup_coef = Proto(0., dtype=float, help="coefficient for supervised loss")

    lr = 1e-3
    weight_decay = 0.0001
    sample_mode = Proto('eps-greedy', help="OneOf[`hard`, `soft`, `eps-greedy`]")
    eps_greedy = Proto(0.05, help="can use a scheduler as in: "
                                  "     `CosineAnneal(0.9, min=0.5, n=num_episodes + 1)`. \n"
                                  "- Always greedy if `True`, \n"
                                  "- Always optimum if `False` or `None`")

    batch_size = Proto(128, dtype=int)
    target_update = Proto(0.9, help="when less than 1, does soft update. When larger or equal to 1, does hard update.")
    replay_memory = Proto(10000, dtype=int)
    prioritized_replay = Proto(False, dtype=bool, help="weighted replay buffer")
    beta = 0.4
    # beta = LinearAnneal(0.4, min=1.0, n=num_episodes + 1)
    latent_dim = Proto(2, help="The dimensionality of the latent space. Set to 2 for easy visualization.")
    matplotlib_backend = Proto(None, help="The backend to use on remote server. Not set by default. Only needed for "
                                          "embedding vis.")

    good_relabel = Proto(False, help="filters the relabled goals with the good_goal from CMaze. ")

    q_fn = Proto('vanilla', dtype=str, help="""the q function to use for the experiment. 
        One of ['vanilla', 'shared-encoder', 'l2-embed', 'l2-embed-T', 'vanilla-conv', 'shared-encoder-conv',
        'l2-embed-conv', 'l2-embed-T-conv'].""")
    embed_p = Proto(2, help="the order of the metric")

    # reporting
    eval_num_envs = 10
    eval_timesteps = 50
    render_num_envs = 10
    metric_summary_interval = 10
    eval_interval = 10
    record_video_interval = 50
    visualize_interval = 10

    # debug options
    embed_id_init = Proto(False)
    embed_id_init_all = Proto(False)

    checkpoint_interval = None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_goal(memory):
    # note: always picks the randomized starting point.
    goal = memory.sample(1)[0].state
    return goal


def optimize_model(memory: ReplayBuffer, policy_net, target_net, optimizer, beta=None):
    if len(memory) < Args.batch_size:
        cprint(f'replay buffer is not ful enough {len(memory)}. Sample again', 'yellow')
        return
    batch = memory.sample(Args.batch_size, beta=beta)

    states = torch.Tensor(batch['s']).to(device)
    actions = torch.LongTensor(batch['a']).to(device)
    next_states = torch.Tensor(batch['s_']).to(device)
    goals = torch.Tensor(batch['g']).to(device)
    rewards = torch.Tensor(batch['r']).to(device)

    if Args.learning_mode == "mdp":
        # note: Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        Q_s = policy_net(states, goals).gather(1, actions[:, None])
        # todo: use value-weighted mean for soft-Q learning
        with torch.no_grad():
            max_Q = target_net(next_states, goals).max(-1)[0]
    else:
        # note: search in neighbors, find embedded neighbor closes to goal.
        #   Standard MDP: Q(s, a, g) = r + max Q(s', a', g)
        #   Passive Learning:
        #                 Q(s, s', g) = r + max Q(s', s", g)
        #   Factorize Q:
        #               - Q(s, s', g) = ||s - s'|| + ||s' - g||
        #   Learning alg:
        #                 ||s - s'|| + ||s' - g|| = r + 1 + min||s" - g||
        #
        # note: we could use soft-Q learning in this case.
        #       states: Size(B, latent_dim) -> (B, None, latent_dim)
        #  memory['z']: Size(N, latent_dim) -> (None, N, latent_dim)
        #       result: Size(B, N).argmax(axis=-1)

        Q_s = policy_net(states, next_states, goals)
        # this is actually the max Q
        with torch.no_grad():
            zs = torch.Tensor(memory['z']).to(device)
            max_Q, _ = target_net.max(states, zs, goals, t=0.1)

    with torch.no_grad():
        # note: Compute V(s_{t+1}) for all next states. # requires_grad=False by default
        non_success_mask = rewards != 0
        # todo: add consistency loss to make Q-function zero when s' = g.
        _ = torch.zeros_like(max_Q, device=device)
        _[non_success_mask] = max_Q[non_success_mask]
        # Compute the expected Q values
        rhs = rewards + Args.gamma * _

    td_error = F.smooth_l1_loss(Q_s.squeeze(), rhs, reduce=False).squeeze()

    # Compute loss
    if Args.prioritized_replay:
        # use the importance sampling ratio as weights.
        q_loss = torch.Tensor(batch['weights']).to(device) * td_error
        new_priorities = q_loss.cpu().data.numpy() + 1e-6
        memory.update_priorities(batch['inds'], new_priorities)
        # this only updates the sampled states
        if Args.learning_mode == 'passive':
            zs = memory.buffer['z']
            for ind, z in zip(batch['inds'], policy_net.embed(states).detach().cpu().numpy()):
                zs[ind] = z
        q_loss = q_loss.mean()
    else:
        q_loss = td_error.mean()

    # todo: add feature flag to turn this On/Off
    # note: forward model consistency loss
    if hasattr(policy_net, 'T'):
        next_state_embeddings = policy_net.next_embed(states, actions)
        next_state_embeddings_hat = policy_net.embed(next_states)
        for_loss1 = F.smooth_l1_loss(
            next_state_embeddings_hat, next_state_embeddings.detach())
        for_loss2 = F.smooth_l1_loss(
            next_state_embeddings, next_state_embeddings_hat.detach())

        with torch.no_grad():
            logger.store_metrics(q_loss=q_loss.cpu().item(), forward_loss=for_loss1,
                                 forward_l2_norm=torch.norm(next_state_embeddings_hat, p=2).item(),
                                 q_values=Q_s.mean().cpu().item(), )

        loss = q_loss + Args.forward_coef * (for_loss1 + for_loss2)
    else:
        with torch.no_grad():
            logger.store_metrics(q_loss=q_loss.cpu().item(),
                                 q_values=Q_s.mean().cpu().item(), )
        loss = q_loss
        # todo: add inverse model loss

    # Supervised regression loss on latent dim and true state and goals
    if Args.sup_coef > 0 and Args.q_fn in ['vanilla-conv', 'shared-encoder-conv']:
        assert Args.latent_dim == 2, "Latent dim must be 2 to use supervised loss"
        s_los = torch.Tensor(batch['s_lo']).to(Args.device)
        g_los = torch.Tensor(batch['g_lo']).to(Args.device)
        sup_loss = F.smooth_l1_loss(policy_net.z, torch.cat([s_los, g_los], dim=-1))
        logger.store_metrics(sup_loss=sup_loss.cpu().item())
        loss += Args.sup_coef * sup_loss

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # todo: remove this, this is bad
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    logger.store_metrics(opt_time=logger.split())


def train(deps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Args.device = device
    Args._update(deps)

    IS_IMAGE = 'Img' in Args.env_id
    obs_shape = [1, 64, 64] if IS_IMAGE else [2]
    if IS_IMAGE:
        lo_obs_shape = [2]

    logger.log_params(Args=vars(Args))

    if Args.matplotlib_backend:
        import matplotlib
        matplotlib.use(Args.matplotlib_backend)
        del matplotlib

    np.random.seed(Args.start_seed)
    torch.manual_seed(Args.start_seed)

    envs = SubprocVecEnv([make_env(Args.env_id, Args.start_seed + i) for i in range(Args.num_envs)])
    logger.log_params(env=envs.spec)

    # use different set of envs from the ones used during training.
    if Args.learning_mode == 'mdp':
        eval_envs = SubprocVecEnv(
            [make_env(Args.env_id, Args.num_envs + Args.start_seed + i) for i in range(Args.eval_num_envs)])
        eval_render_envs = SubprocVecEnv(
            [make_env(Args.env_id, Args.num_envs + Args.start_seed + i) for i in range(Args.render_num_envs)])

    input_dim = envs.observation_space.spaces[Args.obs_key].shape[0]
    act_dim = envs.action_space.n
    num_channels = obs_shape[0]
    if Args.learning_mode == "mdp":
        if Args.q_fn == "vanilla":
            policy_net = QMlp(input_dim * 2, act_dim).to(device)
            target_net = QMlp(input_dim * 2, act_dim).to(device)
        elif Args.q_fn == "shared-encoder":
            policy_net = QSharedEmbedding(input_dim, act_dim, Args.latent_dim).to(device)
            target_net = QSharedEmbedding(input_dim, act_dim, Args.latent_dim).to(device)
        elif Args.q_fn == "l2-embed":
            policy_net = QL2Embed(input_dim, act_dim, Args.latent_dim, Args.embed_p).to(device)
            target_net = QL2Embed(input_dim, act_dim, Args.latent_dim, Args.embed_p).to(device)
        elif Args.q_fn == "l2-embed-T":
            policy_net = QL2EmbedModel(input_dim, act_dim, Args.latent_dim, Args.embed_p).to(device)
            target_net = QL2EmbedModel(input_dim, act_dim, Args.latent_dim, Args.embed_p).to(device)
        elif Args.q_fn == "vanilla-conv":
            policy_net = QConv(num_channels * 2, act_dim, Args.latent_dim).to(device)
            target_net = QConv(num_channels * 2, act_dim, Args.latent_dim).to(device)
        elif Args.q_fn == "vanilla-deep-conv":
            policy_net = QDeepConv(num_channels * 2, act_dim, Args.latent_dim).to(device)
            target_net = QDeepConv(num_channels * 2, act_dim, Args.latent_dim).to(device)
        elif Args.q_fn == "shared-encoder-conv":
            policy_net = QSharedEmbeddingConv(num_channels, act_dim, Args.latent_dim).to(device)
            target_net = QSharedEmbeddingConv(num_channels, act_dim, Args.latent_dim).to(device)
        elif Args.q_fn == "l2-embed-conv":
            policy_net = QL2EmbedConv(num_channels, act_dim, Args.latent_dim, Args.embed_p).to(device)
            target_net = QL2EmbedConv(num_channels, act_dim, Args.latent_dim, Args.embed_p).to(device)
        elif Args.q_fn == "l2-embed-T-conv":
            policy_net = QL2EmbedModelConv(num_channels, act_dim, Args.latent_dim, Args.embed_p).to(device)
            target_net = QL2EmbedModelConv(num_channels, act_dim, Args.latent_dim, Args.embed_p).to(device)
        else:
            Net = globals()[Args.q_fn]
            policy_net = Net(num_channels, act_dim, Args.latent_dim, Args.embed_p).to(device)
            target_net = Net(num_channels, act_dim, Args.latent_dim, Args.embed_p).to(device)
            # raise NotImplementedError(f'{Args.q_fn} is not implemented. Please check your run')
    elif Args.learning_mode == "passive":
        if Args.q_fn == "l2-embed":
            policy_net = QL2EmbeddingPassive(input_dim, Args.latent_dim).to(device)
            target_net = QL2EmbeddingPassive(input_dim, Args.latent_dim).to(device)
        elif Args.q_fn == "l2-embed-conv":
            policy_net = QL2EmbeddingPassiveConv(num_channels, Args.latent_dim).to(device)
            target_net = QL2EmbeddingPassiveConv(num_channels, Args.latent_dim).to(device)
        else:
            raise NotImplementedError(f'{Args.q_fn} is not implemented. Please check your run')
    else:
        raise NotImplementedError(f'{Args.q_fn} is not implemented. Please check your run')

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # make target network eval only

    optimizer = optim.RMSprop(policy_net.parameters(), lr=Args.lr)

    if Args.prioritized_replay:
        memory = PrioritizedReplayBuffer(Args.replay_memory, alpha=0.6)
        # done: need to remove baseline.schedules.
        beta = Args.beta
    else:
        memory = ReplayBuffer(Args.replay_memory)
        beta = None

    from gmo.mdp.sampler import path_gen_fn

    random_pi = lambda ob, goal, *_: np.random.randint(0, act_dim, size=[len(ob)])
    optim_pi = torchify(policy_net.hard, device, dtype=torch.float32)

    eps = Args.eps_greedy

    def greedy_pi(ob, goal):
        nonlocal eps
        if eps is None or eps is False:
            return optim_pi(ob, goal)
        if eps is True or np.random.rand(1) < eps:
            return random_pi(ob, goal)
        return optim_pi(ob, goal)

    # fn = policy_net.d if hasattr(policy_net, 'd') else lambda x, g: policy_net(x, g)[:, 0].detach()

    if IS_IMAGE:
        #     Q_embed_2d = imaginator(envs, fn, width=obs_shape[1], height=obs_shape[2], device=device)
        #
        #     if hasattr(policy_net, 'embed'):
        #         viz_embed_2d = imaginator(envs, policy_net.embed, device=device)
        all_keys = ('x', 'goal', 'img', 'goal_img')
    else:
        #     Q_embed_2d = torchify(fn, device, dtype=torch.float32)
        #
        #     if hasattr(policy_net, 'embed'):
        #         viz_embed_2d = torchify(policy_net.embed, device, dtype=torch.float32)
        all_keys = tuple()

    greedy_path_gen = path_gen_fn(envs, greedy_pi, Args.obs_key, Args.goal_key, all_keys=all_keys,
                                  num_batch_cap=None if Args.n_rollouts is None else Args.n_rollouts // Args.num_envs)

    next(greedy_path_gen)

    if Args.learning_mode == "mdp":
        # eval_gen = sample_gen_fn(eval_envs, optim_pi, Args.obs_key, Args.goal_key)
        # eval_render_gen = render_gen_fn(eval_render_envs, optim_pi, Args.obs_key, Args.goal_key, width=64, height=64)
        eval_gen = sample_gen_fn(eval_envs, greedy_pi, Args.obs_key, Args.goal_key)
        eval_render_gen = render_gen_fn(eval_render_envs, greedy_pi, Args.obs_key, Args.goal_key, width=64, height=64)

    steps_done = 0

    t0 = time.time()
    for ep_ind in range(0, Args.num_episodes + 1):
        logger.store_metrics(dt_epoch=time.time() - t0)
        t0 = time.time()
        # Do all the logging and visualization upfront
        start_time = time.time()
        # if Args.visualize_interval and ep_ind % Args.visualize_interval == 0:
        #     cprint('visualize Q-functions in 7x7', 'green')
        #     with torch.no_grad():
        #         visualize_q_2d(Q_embed_2d, goal_n=7, key=f"figures/q_{ep_ind:05d}.png")
        #         # for a in range(envs.action_space.n):
        #         #     visualize_q_2d(f"q_{ep_ind:05d}_{a}", torchify(lambda *args: policy_net(*args)[:, a], device), goal_n=7)
        #
        #     # if hasattr(policy_net, 'embed') and Args.latent_dim == 2:
        #     #     cprint('visualize embedding function', 'green')
        #     #     with torch.no_grad():
        #     #         visualize_embedding_2d(f'embed_{ep_ind:05d}', viz_embed_2d)

        if ep_ind % Args.eval_interval == 0:

            if Args.learning_mode == 'mdp':
                policy_net.eval()
                paths = pandas.DataFrame([next(eval_gen) for _ in range(Args.eval_timesteps)])
                success_rate = (paths['info.success'] * paths['new']).sum() / paths['new'].sum()
                logger.store_metrics({"eval/success_rate": success_rate, "eval/distance": paths['info.dist'].mean(),
                                      "eval/rewards": paths['rew'].mean()})
                policy_net.train()

        if ep_ind and ep_ind % Args.metric_summary_interval == 0:
            logger.log_metrics_summary(key_values=dict(timesteps=steps_done, episode=ep_ind, eps=eps, beta=beta),
                                       default_stats='mean')

        if Args.record_video_interval and ep_ind % Args.record_video_interval == 0 and Args.learning_mode == 'mdp':
            policy_net.eval()
            paths = pandas.DataFrame([next(eval_render_gen) for _ in range(Args.eval_timesteps)])
            logger.log_video(np.swapaxes(np.stack(paths['view']), 0, 1).reshape(-1, 64, 64, 3),
                             f"videos/{Args.env_id}_{ep_ind:05d}.mp4", fps=30)
            success_rate = (paths['info.success'] * paths['new']).sum() / paths['new'].sum()
            logger.store_metrics({"eval/success_rate": success_rate, "eval/distance": paths['info.dist'].mean(),
                                  "eval/rewards": paths['rew'].mean()})
            policy_net.train()

        metrics_time = time.time()
        logger.store_metrics(metrics_time=metrics_time - start_time)

        # sampling starts here.
        with torch.no_grad():
            if Args.learning_mode == "passive":
                if ep_ind == 0:
                    raise NotImplementedError("todo: to fix this")
                    p = rand_path_gen.send(Args.batch_timesteps)
                    # add visualization of the trajectories here. What is the shape for the trajectories?
                    # note: won't happen here for active version.
                    print(p.keys())

            if isinstance(Args.eps_greedy, Schedule):
                eps = Args.eps_greedy.send(ep_ind)

            if Args.prioritized_replay and isinstance(Args.beta, Schedule):
                beta = Args.beta.send(ep_ind)

            # epsilon greedy
            p = greedy_path_gen.send(Args.batch_timesteps)

        steps_done += Args.batch_timesteps * Args.num_envs

        if Args.learning_mode == "mdp" or ep_ind == 0:
            states = p['obs'][Args.obs_key]

            if Args.learning_mode == "passive":
                # zs: Size(H, N, W).
                zs = policy_net.embed(torch.Tensor(states.reshape(-1, *obs_shape)).to(device)) \
                    .reshape(*states.shape[:2], Args.latent_dim).detach().cpu().numpy()
                _ = {'z': zs.reshape(-1, Args.latent_dim)} if Args.learning_mode == "passive" else {}
            else:
                _ = {}

            # passive mode: need to supply keys
            # Actual sampled trajectories.
            if IS_IMAGE:
                memory.extend(s=states.reshape(-1, *obs_shape), a=p['acs'].reshape(-1),
                              r=p['rewards'].reshape(-1), s_=p['next'][Args.obs_key].reshape(-1, *obs_shape),
                              g=p['obs'][Args.goal_key].reshape(-1, *obs_shape),
                              s_lo=p['obs']['x'].reshape(-1, *lo_obs_shape),
                              g_lo=p['obs']['goal'].reshape(-1, *lo_obs_shape), **_)
            else:
                memory.extend(s=states.reshape(-1, *obs_shape), a=p['acs'].reshape(-1),
                              r=p['rewards'].reshape(-1), s_=p['next'][Args.obs_key].reshape(-1, *obs_shape),
                              g=p['obs'][Args.goal_key].reshape(-1, *obs_shape), **_)
            # note: for debug, use sampled_p = memory.sample(Args.optim_batch_size)

            success_rate = (p['info.successes'] * p['dones']).sum() / p['dones'].sum()
            logger.store_metrics(rewards=p['rewards'].mean(), success_rate=success_rate)

            # Relabeled trajectories
            # note: Three relabel strategies were mentioned in the HER paper:
            #  "future", "episode", "random". Here we implement the "episode" strategy.
            #       we can not use the p['next'] states because they contain None states
            #  we could use original goal as what Amy did, but the time-limited environments
            #  contain terminations that are not successful.
            #       Therefore, we use the original states obs['x'], instead.
            assert Args.her_strategy == "episode", "only `episode` version is implemented."
            k = Args.batch_timesteps // Args.her_k_goals
            act_mask = np.array(1 - p['dones'], dtype=bool)
            if IS_IMAGE:
                goal_los = p['obs']['goal']
                mask = np.broadcast_to(act_mask[:, :, None, None, None], states.shape).copy()
                next_mask = np.broadcast_to(act_mask[:, :, None, None, None], states.shape).copy()
                x = p['obs']['x']
                mask_low = np.broadcast_to(act_mask[:, :, None], x.shape).copy()
                next_mask_low = np.broadcast_to(act_mask[:, :, None], x.shape).copy()
                next_mask_low[1:] = next_mask_low[:-1]
                mask_low[-1] = next_mask_low[0] = False
                x_masked = x[next_mask_low].reshape(-1, *lo_obs_shape)
                new_gs = p['obs']['x'][k - 1::k]
            else:
                mask = np.broadcast_to(act_mask[:, :, None], states.shape).copy()  # you will know why
                next_mask = np.broadcast_to(act_mask[:, :, None], states.shape).copy()  # if you don't ;)

            next_mask[1:] = next_mask[:-1]  # shift the mask to the right by 1 timesteps
            act_mask[-1] = mask[-1] = next_mask[0] = False

            new_goals = states[k - 1::k]
            for idx, goals in enumerate(new_goals):
                if Args.good_relabel:
                    from ge_world.c_maze import good_goal
                    goal_mask_a = np.array([good_goal(g) for g in (goal_los if IS_IMAGE else goals)])
                else:
                    goal_mask_a = np.array([True for g in range(len(goals))])

                goal_mask_lo = goal_mask_a[:, None]
                goal_mask = goal_mask_a[:, None, None, None] if IS_IMAGE else goal_mask_lo

                if IS_IMAGE:
                    goal_los = new_gs[idx]
                    goal_los = np.broadcast_to(goal_los[None, :, :], x.shape)[
                        next_mask_low * goal_mask_lo].reshape(-1, *lo_obs_shape)

                new_goal = np.broadcast_to(goals[None, :, :], states.shape)[
                    next_mask * goal_mask].reshape(-1, *obs_shape)

                s = states[mask * goal_mask].reshape(-1, *obs_shape)
                s_next = states[next_mask * goal_mask].reshape(-1, *obs_shape)
                _ = mask_low if IS_IMAGE else mask
                if Args.learning_mode == "passive":
                    z_dict = {'z': zs[_ * goal_mask_lo].reshape(-1, Args.latent_dim)}
                else:
                    z_dict = {}

                if IS_IMAGE:
                    new_rew = envs.first_call_sync('compute_reward', x_masked, goal_los, None)
                    memory.extend(s=s, a=p['acs'][act_mask * goal_mask_a], r=new_rew, s_=s_next, g=new_goal,
                                  s_lo=x[mask_low * goal_mask_lo].reshape(-1, *lo_obs_shape),
                                  g_lo=goal_los, **z_dict)
                else:
                    new_rew = envs.first_call_sync('compute_reward', s_next, new_goal, None)
                    memory.extend(s=s, a=p['acs'][act_mask * goal_mask_a], r=new_rew, s_=s_next,
                                  g=new_goal, **z_dict)

        logger.split()
        for _ in range(Args.optim_steps):
            optimize_model(memory, policy_net, target_net, optimizer, beta)

            if Args.target_update < 1:  # <-- soft target update
                q_params = policy_net.state_dict()
                with torch.no_grad():
                    for name, p in target_net.state_dict().items():
                        p.copy_((1 - Args.target_update) * q_params[name] + Args.target_update * p)
            elif (ep_ind * Args.optim_steps + _) % Args.target_update == 0:  # <-- hard target update
                target_net.load_state_dict(policy_net.state_dict())

        if Args.checkpoint_interval and ep_ind % Args.checkpoint_interval == 0:
            logger.remove(f"models/{ep_ind - Args.checkpoint_interval:05d}-policy_net.pkl")
            path = logger.save_module(policy_net, f"models/{ep_ind:05d}-policy_net.pkl", chunk=200_000_000)

    if Args.learning_mode == "mdp":
        eval_gen.close()
        eval_render_gen.close()

    greedy_path_gen.close()


if __name__ == "__main__":
    from plan2vec_experiments import instr, config_charts

    with logger.SyncContext(clean=True):  # single use Synchronous Context
        config_charts("""
        charts:
          - {glob: '**/*.png', type: file}
          - yDomain: [0, 1]
            yKey: success_rate/mean
            xKey: episode
          - {glob: '**/*.mp4', type: video}
        keys:
          - run.status
          - Args.n_rollouts
          - DEBUG.ground_truth_neighbor_r
          - DEBUG.supervised_value_fn
          - Args.term_r
        """)
        instr(train)()
