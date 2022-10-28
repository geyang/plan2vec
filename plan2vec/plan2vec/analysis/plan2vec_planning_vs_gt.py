from matplotlib import pyplot as plt


def maze_plan_vs_gt(search_alg, seed, env_id, num_rollouts):
    from graph_search import methods
    import pandas as pd
    from plan2vec.plan2vec.maze_plan2vec import sample_trajs, MazeGraph, Args
    from tqdm import trange
    from ml_logger import logger
    import numpy as np

    trajs = sample_trajs(seed, env_id=env_id, num_rollouts=num_rollouts,
                         obs_keys=["x"], limit=2)
    logger.print('finished sampling', color="green")

    graph = MazeGraph(trajs, obs_keys=["x"], r=Args.neighbor_r, r_min=Args.neighbor_r_min)

    graph.show(f"figures/{env_id}/samples.png", show=False)

    search = methods[search_alg]

    logger.summary_cache.clear()

    for i in trange(100):
        start, goal = graph.sample(2)
        gt_distance = np.linalg.norm(graph.nodes[start]['pos'] - graph.nodes[goal]['pos'], ord=2)
        try:
            path, ds = search(graph, start, goal)
            path_length = np.sum(ds)
            logger.store_metrics(l2=gt_distance, path_length=path_length)
        except:
            pass

    colors = ['#49b8ff', '#ff7575', '#66c56c', '#f4b247']
    data = pd.DataFrame(logger.summary_cache.data)

    plt.figure(figsize=(3, 3), dpi=200)
    plt.title(search.__doc__.split('\n')[0])
    plt.scatter(data['l2'], data['path_length'], marker="o", color=colors[0])
    plt.ylabel("Planned Path Length")
    plt.xlabel("Ground-truth L2 Distance")
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    logger.savefig(f'./figures/{env_id}/{search_alg}_plan_length_vs_gt.png')
    # plt.show()


def streetlearn_plan_vs_gt(search_alg, env_id, p):
    from graph_search import methods
    import pandas as pd
    from tqdm import trange
    from ml_logger import logger
    import numpy as np
    from plan2vec.plan2vec.streetlearn_plan2vec import load_streetlearn, StreetLearnGraph

    dataset, start, goal = load_streetlearn(f"~/fair/streetlearn/processed-data/{env_id}")
    graph = StreetLearnGraph(dataset, r=2.4e-4)
    del dataset

    graph.show(f"figures/{env_id}/samples.png", show=False)

    search = methods[search_alg]

    logger.summary_cache.clear()

    for i in trange(300):
        start, goal = graph.sample(2)
        gt_distance = np.linalg.norm(graph.nodes[start]['pos'] - graph.nodes[goal]['pos'], ord=p)
        try:
            path, ds = search(graph, start, goal)
            path_length = np.sum(ds)
            logger.store_metrics(l2=gt_distance, path_length=path_length)
        except:
            pass

    colors = ['#49b8ff', '#ff7575', '#66c56c', '#f4b247']
    data = pd.DataFrame(logger.summary_cache.data)

    name = search.__doc__.split('\n')[0].split(" ")[0]

    plt.figure(figsize=(4, 3), dpi=200)
    plt.title(env_id)
    plt.plot(data['l2'], data['path_length'], marker="o", linewidth=0, color=colors[0], alpha=0.7,
             markeredgecolor="none",
             label=name)
    plt.ylabel("Planned Path Length")
    plt.xlabel(f"Ground-truth L{p} Distance")
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(bbox_to_anchor=(0.75, 0.4), framealpha=1, frameon=False, fontsize=8)
    plt.tight_layout()
    logger.savefig(f'./figures/{env_id}/{search_alg}_plan_length_vs_gt_L{p}.png')
    plt.show()


if __name__ == '__main__':
    # for env_id in ["manhattan-tiny", "manhattan-medium", "manhattan-large"]:
    #     for search_alg in ['bfs', "dijkstra"]:
    #         for p in [1, 2]:
    #             streetlearn_plan_vs_gt(search_alg, env_id, p)

    for env_id in ['GoalMassDiscreteIdLess-v0', 'CMazeDiscreteIdLess-v0']:
        for search_alg in ['bfs', "dijkstra"]:
            maze_plan_vs_gt(search_alg, seed=10, env_id=env_id, num_rollouts=200, )

    exit()

if __name__ == '__main__':
    import jaynes
    from plan2vec_experiments import instr

    jaynes.config('cpu')

    for env_id in ['GoalMassDiscreteIdLess-v0', 'CMazeDiscreteIdLess-v0']:
        for search_alg in ['bfs', "dijkstra"]:
            thunk = instr(maze_plan_vs_gt, search_alg,
                          seed=10, env_id=env_id, num_rollouts=200,
                          __postfix="planning_analysis", __no_timestamp=True)
            jaynes.run(thunk)

    jaynes.listen()
