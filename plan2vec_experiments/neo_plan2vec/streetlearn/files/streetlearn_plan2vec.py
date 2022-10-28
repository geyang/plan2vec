"""
New Plan2vec Implementation, uses graph-planning algorithms to navigate the graph.

Now adapted for StreetLearn

Test if the value function blows up or not.
"""
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from params_proto.neo_proto import ParamsProto, Proto
import torch

from plan2vec.mdp.replay_buffer import EpisodicBuffer, ReplayBuffer
from plan2vec.mdp.td_lambda import mc_target

from streetlearn import StreetLearnDataset

# to scale the various datasets
# Info: this is wrong, need to be recalculated.
from torch_utils import Eval

# Info: this is wrong.
r_scale_dict = {
    "manhattan-tiny": 0.2,
    "manhattan-small": 0.09,
    "manhattan-medium": 0.06,
    "manhattan-large": 0.04,
    "manhattan-xl": 0.02,
}


class Args(ParamsProto):
    seed = 10

    env_id = "manhattan-tiny"
    input_dim = 1
    latent_dim = 2
    metric_p = 1
    view_mode = "omni-gray"

    # sample parameters
    n_envs = 10
    num_rollouts = 200
    limit = 2  # limit for the timestamp.

    # graph parameters
    neighbor_r = 2.4e-4
    neighbor_r_min = None

    # search params
    h_scale = 1.2
    search_alg = "a_star"

    num_epochs = 800
    batch_size = 10  # sampling batch_size

    # learning parameters
    optim_epochs = 10
    optim_batch_size = 100
    lr = 1e-5
    lam = 0.8
    gamma = 0.99
    r_scale = Proto(1, help="scaling factor for the reward")
    global_metric = "ResNet18L2"
    target_update = False

    device = None
    criteria = Proto("nn.MSELoss(reduction='mean')", help="evaluated to a criteria object")

    visualization_interval = 10
    checkpoint_interval = 500
    checkpoint_overwrite = True


class DEBUG(ParamsProto):
    supervise_value = False


def load_streetlearn(data_path="~/fair/streetlearn/processed-data/manhattan-large", pad=0.1):
    import matplotlib.pyplot as plt
    from os.path import expanduser
    path = expanduser(data_path)
    d = StreetLearnDataset(path, (64, 64), Args.view_mode)
    d.select_all()

    a = d.bbox[0] + d.bbox[2] * pad, d.bbox[1] + d.bbox[3] * pad
    b = d.bbox[0] + d.bbox[2] * (1 - pad), d.bbox[1] + d.bbox[3] * (1 - pad)
    (start, _), (goal, _) = d.locate_closest(*a), d.locate_closest(*b)

    # fig = plt.figure(figsize=(6, 5))
    # plt.scatter(*d.lng_lat[start], marker="o", s=100, linewidth=3,
    #             edgecolor="black", facecolor='none', label="start")
    # plt.scatter(*d.lng_lat[goal], marker="x", s=100, linewidth=3,
    #             edgecolor="none", facecolor='red', label="end")
    # plt.legend(loc="upper left", bbox_to_anchor=(0.95, 0.7), framealpha=1,
    #            frameon=False, fontsize=12)
    # d.show_blowout("NYC-large", fig=fig, box_color='gray', box_alpha=0.1,
    #                show=True, set_lim=True)

    return d, start, goal


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


class StreetLearnGraph(nx.Graph):
    def __init__(self, dataset: StreetLearnDataset, r, p=1):
        super().__init__()
        from tqdm import tqdm

        self.all_positions = dataset.lng_lat
        if len(dataset.images.shape) == 3:
            self.all_images = dataset.images[:, None, ...] / 255
        elif len(dataset.images.shape) == 4:
            self.all_images = dataset.images.transpose(0, 3, 1, 2) / 255
        else:
            raise NotImplementedError(f"image tensor dimension {dataset.images.shape} is not supported.")
        self.lat_correction = dataset.lat_correction

        for node, xy in enumerate(tqdm(self.all_positions, desc="build graph")):
            self.add_node(node, pos=xy)

        for node, a in tqdm(self.nodes.items(), desc="add edges"):
            (ll,), (ds,), (ns,) = dataset.neighbor([node], r=r, ord=p)
            for neighbor, d in zip(ns, ds):
                self.add_edge(node, neighbor, weight=d)

        self.queries = defaultdict(lambda: 0)

    def pop_cost(self):
        cost = len(self.queries.keys())
        self.queries.clear()
        return cost

    def neighbors(self, n):
        ns = list(super().neighbors(n))
        for n in ns:
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
        plt.gca().set_aspect(self.lat_correction)
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

    def plot_traj(self, path, title=None, show=True, color='black', **kwargs):
        if show:
            plt.figure(figsize=(2, 2), dpi=120)
        if title:
            plt.title(title)
        for (x, y), (x_, y_) in zip(path[:-1], path[1:]):
            dx = (x_ - x)
            dy = (y_ - y)
            d = np.linalg.norm([dx, dy], ord=2)
            plt.arrow(x, y, dx * 0.8, dy * 0.8, **kwargs, head_width=d * 0.3, head_length=d * 0.3,
                      length_includes_head=True, head_starts_at_zero=True, fc=color, ec=color)

        plt.gca().set_aspect(self.lat_correction)
        if show:
            plt.show()


def train(deps=None):
    from graph_search import methods
    from ml_logger import logger
    from torch import optim, nn
    from torch_utils import torchify
    from tqdm import trange
    from plan2vec.models.resnet import ResNet18CoordL2, ResNet18L2, ResNet18Kernel
    from plan2vec.plotting.streetlearn.embedding_image_streetlearn \
        import cache_images, visualize_embedding_2d_image, visualize_embedding_3d_image

    Args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Args._update(deps)
    DEBUG._update(deps)

    np.random.seed(Args.seed)
    torch.manual_seed(Args.seed)
    # see https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.log_params(Args=vars(Args), DEBUG=vars(DEBUG))

    logger.upload_file(__file__)  # upload this file to the experiment folder

    dataset, start, goal = load_streetlearn(f"~/fair/streetlearn/processed-data/{Args.env_id}")
    graph = StreetLearnGraph(dataset, Args.neighbor_r, p=Args.metric_p)
    del dataset
    graph.show("figures/sample_graph.png")

    search = methods[Args.search_alg]

    Φ = eval(Args.global_metric)(
        Args.input_dim, latent_dim=Args.latent_dim, p=Args.metric_p).to(Args.device)

    d = torchify(Φ, dtype=torch.float32, device=Args.device, input_only=True)
    value_fn = lambda *_: - d(*_).squeeze()
    heuristic = lambda a, b: d(graph.all_images[a], graph.all_images[b]).squeeze().cpu().numpy() * Args.h_scale

    # @torchify(dtype=torch.float32)
    def embed_eval(*_):
        with Eval(Φ), torch.no_grad():
            return Φ.embed(*_)  # .squeeze()

    criteria = eval(Args.criteria)
    optimizer = optim.Adam(Φ.parameters(), lr=Args.lr, )

    # graph.plot_traj(path, title="CMaze")

    logger.print('start learning', color="yellow")
    # Implement multi-step value function
    # todo: test the target value vs ground-truth
    buffer = EpisodicBuffer(10000)
    diagnosis_buffer = ReplayBuffer(10000)
    for epoch in range(Args.num_epochs + 1):
        # sampling from the graph
        for i in trange(Args.batch_size, desc="sample"):
            while True:
                try:
                    # todo: add curriculum for goal setting. 5 lines of code.
                    # if Args.search_alg == 'a_star': # this requires curriculum
                    #     while (graph.all_positions[start, goal] * magic).norm(axis=-1) < epoch / Args.num_epochs:
                    #         start, goal = graph.sample(2)
                    start, goal = graph.sample(2)
                    with torch.no_grad():
                        path, ds = search(graph, start, goal, heuristic)
                    logger.store_metrics(cost=graph.pop_cost(), path_length=np.sum(ds), plan_steps=len(ds))
                    buffer.extend(start=start, goal=goal, path=path, ds=np.array(ds) * Args.r_scale, )
                    break
                except Exception as e:
                    print(f'planning failed {start}, {goal}')
                    pass

        for i in trange(Args.optim_epochs, desc="optim"):
            samples = buffer.sample(Args.optim_batch_size)

            states = graph.all_images[[traj['path'][0] for traj in samples]]
            goal_states = graph.all_images[[traj['goal'] for traj in samples]]

            target_values = torch.tensor([
                -mc_target(traj['ds']) for traj in samples],
                dtype=torch.float32, device=Args.device)

            if DEBUG.supervise_value:
                starts = torch.tensor(graph.all_positions[[traj['start'] for traj in samples]],
                                      dtype=torch.float32, device=Args.device)
                goals = torch.tensor(graph.all_positions[[traj['goal'] for traj in samples]],
                                     dtype=torch.float32, device=Args.device)
                next_states_pos = torch.tensor(graph.all_positions[[traj['path'][1] for traj in samples]],
                                               dtype=torch.float32, device=Args.device)
                magic = torch.tensor([1, graph.lat_correction], dtype=torch.float32, device=Args.device)
                target_values = - torch.norm((starts - goals) * magic, p=Args.metric_p, dim=-1) * Args.r_scale

            values = value_fn(states, goal_states)
            # values_2 = value_fn(states, next_states)
            loss = criteria(values, target_values)  # + criteria(values_2, target_values_2)
            loss.backward()
            optimizer.step()

            logger.store_metrics(loss=loss.cpu().item(), value=values.mean().cpu().item(),
                                 target_values=target_values.mean().item(), )

            # info: can remove if not needed.
            diagnosis_buffer.extend(values=values.detach().cpu().numpy(),
                                    target_values=target_values.detach().cpu().numpy())

        logger.log_metrics_summary(key_values=dict(epoch=epoch, dt_epoch=logger.split()))

        if Args.visualization_interval and epoch % Args.visualization_interval == 0:
            cache_images(graph.all_images, graph.all_positions, graph.lat_correction)
            eval(f"visualize_embedding_{Args.latent_dim}d_image")(  # png version for quick view
                Args.env_id, embed_eval, f"figures/embedding/embed_{Args.latent_dim}d_{epoch:05d}.png")

            # info: can remove if not needed.
            import matplotlib.pyplot as plt
            plt.figure(figsize=(3.2, 3), dpi=200)

            plt.title(f"Prediction vs GT L{Args.metric_p}")
            plt.xlabel('target_values')
            plt.ylabel('predicted values')
            plt.gca().set_aspect('equal', 'datalim')
            plt.scatter(diagnosis_buffer['target_values'],
                        diagnosis_buffer['values'],
                        color="#23aaff")
            plt.tight_layout()
            logger.savefig(f"debug/{epoch:04d}_pred_vs_gt.png")
            plt.close()
            diagnosis_buffer.clear()

        if Args.checkpoint_interval and epoch % Args.checkpoint_interval == 0:
            logger.save_module(Φ, f"models/{epoch:05d}-Φ.pkl", chunk=200_000_000)
            if Args.checkpoint_overwrite:
                logger.remove(f"models/{epoch - Args.checkpoint_interval:05d}-Φ.pkl")


class EvalArgs(ParamsProto):
    seed = 10

    env_id = "manhattan-tiny"
    input_dim = 1
    latent_dim = 2
    metric_p = 1
    view_mode = "omni-gray"

    # sample parameters
    n_envs = 10
    num_rollouts = 200
    limit = 2  # limit for the timestamp.

    # graph parameters
    neighbor_r = 2.4e-4
    neighbor_r_min = None

    # search params
    h_scale = 1.2
    search_alg = "a_star"

    num_epochs = 800
    batch_size = 10  # sampling batch_size

    global_metric = "ResNet18L2"

    device = None


def planning_eval(deps=None):
    from graph_search import methods
    from ml_logger import logger
    from torch import optim, nn
    from torch_utils import torchify
    from tqdm import trange
    from plan2vec.models.resnet import ResNet18CoordL2, ResNet18L2, ResNet18Kernel
    from plan2vec.plotting.streetlearn.embedding_image_streetlearn \
        import cache_images, visualize_embedding_2d_image, visualize_embedding_3d_image

    EvalArgs.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EvalArgs._update(deps)
    DEBUG._update(deps)

    np.random.seed(EvalArgs.seed)
    torch.manual_seed(EvalArgs.seed)
    # see https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.log_params(EvalArgs=vars(EvalArgs), DEBUG=vars(DEBUG))

    logger.upload_file(__file__)  # upload this file to the experiment folder

    dataset, start, goal = load_streetlearn(f"~/fair/streetlearn/processed-data/{EvalArgs.env_id}")
    graph = StreetLearnGraph(dataset, EvalArgs.neighbor_r, p=EvalArgs.metric_p)
    del dataset
    graph.show("figures/sample_graph.png")

    search = methods[EvalArgs.search_alg]

    Φ = eval(EvalArgs.global_metric)(
        EvalArgs.input_dim, latent_dim=EvalArgs.latent_dim, p=EvalArgs.metric_p).to(EvalArgs.device)

    d = torchify(Φ, dtype=torch.float32, device=EvalArgs.device, input_only=True)
    value_fn = lambda *_: - d(*_).squeeze()
    heuristic = lambda a, b: d(graph.all_images[a], graph.all_images[b]).squeeze().cpu().numpy() * EvalArgs.h_scale

    # @torchify(dtype=torch.float32)
    def embed_eval(*_):
        with Eval(Φ), torch.no_grad():
            return Φ.embed(*_)  # .squeeze()

    criteria = eval(EvalArgs.criteria)
    optimizer = optim.Adam(Φ.parameters(), lr=EvalArgs.lr, )

    # graph.plot_traj(path, title="CMaze")

    logger.print('start learning', color="yellow")
    # Implement multi-step value function
    # todo: test the target value vs ground-truth
    buffer = EpisodicBuffer(10000)
    diagnosis_buffer = ReplayBuffer(10000)
    for epoch in range(EvalArgs.num_epochs + 1):
        # sampling from the graph
        for i in trange(EvalArgs.batch_size, desc="sample"):
            while True:
                try:
                    # todo: add curriculum for goal setting. 5 lines of code.
                    # if EvalArgs.search_alg == 'a_star': # this requires curriculum
                    #     while (graph.all_positions[start, goal] * magic).norm(axis=-1) < epoch / EvalArgs.num_epochs:
                    #         start, goal = graph.sample(2)
                    start, goal = graph.sample(2)
                    with torch.no_grad():
                        path, ds = search(graph, start, goal, heuristic)
                    logger.store_metrics(cost=graph.pop_cost(), path_length=np.sum(ds), plan_steps=len(ds))
                    buffer.extend(start=start, goal=goal, path=path, ds=np.array(ds) * EvalArgs.r_scale, )
                    break
                except Exception as e:
                    print(f'planning failed {start}, {goal}')
                    pass

        for i in trange(EvalArgs.optim_epochs, desc="optim"):
            samples = buffer.sample(EvalArgs.optim_batch_size)

            states = graph.all_images[[traj['path'][0] for traj in samples]]
            goal_states = graph.all_images[[traj['goal'] for traj in samples]]

            target_values = torch.tensor([
                -mc_target(traj['ds']) for traj in samples],
                dtype=torch.float32, device=EvalArgs.device)

            if DEBUG.supervise_value:
                starts = torch.tensor(graph.all_positions[[traj['start'] for traj in samples]],
                                      dtype=torch.float32, device=EvalArgs.device)
                goals = torch.tensor(graph.all_positions[[traj['goal'] for traj in samples]],
                                     dtype=torch.float32, device=EvalArgs.device)
                next_states_pos = torch.tensor(graph.all_positions[[traj['path'][1] for traj in samples]],
                                               dtype=torch.float32, device=EvalArgs.device)
                magic = torch.tensor([1, graph.lat_correction], dtype=torch.float32, device=EvalArgs.device)
                target_values = - torch.norm((starts - goals) * magic, p=EvalArgs.metric_p, dim=-1) * EvalArgs.r_scale

            values = value_fn(states, goal_states)
            # values_2 = value_fn(states, next_states)
            loss = criteria(values, target_values)  # + criteria(values_2, target_values_2)
            loss.backward()
            optimizer.step()

            logger.store_metrics(loss=loss.cpu().item(), value=values.mean().cpu().item(),
                                 target_values=target_values.mean().item(), )

            # info: can remove if not needed.
            diagnosis_buffer.extend(values=values.detach().cpu().numpy(),
                                    target_values=target_values.detach().cpu().numpy())

        logger.log_metrics_summary(key_values=dict(epoch=epoch, dt_epoch=logger.split()))

        if EvalArgs.visualization_interval and epoch % EvalArgs.visualization_interval == 0:
            cache_images(graph.all_images, graph.all_positions, graph.lat_correction)
            eval(f"visualize_embedding_{EvalArgs.latent_dim}d_image")(  # png version for quick view
                EvalArgs.env_id, embed_eval, f"figures/embedding/embed_{EvalArgs.latent_dim}d_{epoch:05d}.png")

            # info: can remove if not needed.
            import matplotlib.pyplot as plt
            plt.figure(figsize=(3.2, 3), dpi=200)

            plt.title(f"Prediction vs GT L{EvalArgs.metric_p}")
            plt.xlabel('target_values')
            plt.ylabel('predicted values')
            plt.gca().set_aspect('equal', 'datalim')
            plt.scatter(diagnosis_buffer['target_values'],
                        diagnosis_buffer['values'],
                        color="#23aaff")
            plt.tight_layout()
            logger.savefig(f"debug/{epoch:04d}_pred_vs_gt.png")
            plt.close()
            diagnosis_buffer.clear()


if __name__ == '__main__':
    import jaynes
    from params_proto.neo_hyper import Sweep
    from plan2vec_experiments import instr, config_charts
    from ml_logger import logger

    jaynes.config()

    # ENVS = ["manhattan-tiny", "manhattan-small", "manhattan-medium", "manhattan-large"]
    ENVS = ["manhattan-tiny", "manhattan-small", "manhattan-large"]

    with Sweep(Args, DEBUG) as sweep:

        DEBUG.supervise_value = True

        Args.num_epochs = 5000
        Args.metric_p = 1
        Args.lr = 3e-7
        Args.batch_size = 100
        Args.global_metric = "ResNet18L2"
        with sweep.product:
            with sweep.zip:
                Args.r_scale = [r_scale_dict[env] for env in ENVS]
                Args.env_id = ENVS
            Args.search_alg = ["dijkstra", "a_star", ]

    for deps in sweep:
        thunk = instr(train, deps, __prefix="streetlearn-supervise",
                      __postfix=f"{Args.env_id}-{Args.search_alg}-{Args.global_metric}")
        jaynes.run(thunk, )
        config_charts("""
        charts:
        - yKey: loss/mean
          xKey: epoch
        - yKey: value/mean
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
