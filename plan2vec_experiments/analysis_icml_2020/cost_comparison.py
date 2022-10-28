from plan2vec_experiments.analysis_icml_2020 import stylize, plot_bar

if __name__ == '__main__':
    """
       bfs len: 123 cost: 1470
 heuristic len: 124 cost: 195
  dijkstra len: 123 cost: 1470
    a_star len: 124 cost: 342
    """

    cache = {
        "ours": 123,
        "Dijkstra's": 1470,
        "BFS": 1470,
    }

    stylize()
    plot_bar(cache.keys(), cache.values(), "Street Learn",
             xlabel=' ',
             filename="figures/cost_comparison_sl.png")
