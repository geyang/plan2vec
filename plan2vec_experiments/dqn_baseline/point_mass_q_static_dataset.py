if __name__ == '__main__':
    import jaynes
    import numpy as np

    from plan2vec.dqn.point_mass_q_learning import Args, train
    from plan2vec_experiments import instr, config_charts
    from params_proto.neo_hyper import Sweep

    jaynes.config('vector-cpu')

    with Sweep(Args) as sweep:
        Args.num_envs = 1
        Args.num_episodes = 2000

        with sweep.product:
            Args.n_rollouts = 20 * np.array([10, 20, 30, 40, 50, 100, 200, 300, 400])

    for deps in sweep:
        jaynes.run(instr(train,
                         deps,
                         __postfix=f"dqn-n_rollouts-({Args.eps_greedy})"))

        config_charts("""
        charts:
          - {glob: '**/*.png', type: file}
          - yDomain: [0, 1]
            yKey: success_rate/mean
            xKey: episode
          - yDomain: [0, 1]
            yKey: eval/success_rate/mean
            xKey: episode
          - {glob: '**/*.mp4', type: video}
        keys:
          - run.status
          - Args.n_rollouts
          - Args.num_envs
          - Args.eps_greedy
          - DEBUG.ground_truth_neighbor_r
          - DEBUG.supervised_value_fn
          - Args.term_r
        """)

    jaynes.listen()
