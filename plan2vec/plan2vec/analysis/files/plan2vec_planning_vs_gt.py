from matplotlib import pyplot as plt

from plan2vec.plan2vec.neo_plan2vec import sample_trajs, MazeGraph, Args


def maze_plan_vs_gt(search_alg, seed, env_id, num_rollouts):
    from graph_search import methods
    import pandas as pd
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
    logger.savefig(f'./figures/{env_id}/{search_alg}_plan_length_vs_gt.png')
    # plt.tight_layout()
    # plt.show()


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
