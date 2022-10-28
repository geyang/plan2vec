if __name__ == '__main__':
    import jaynes
    from params_proto.neo_hyper import Sweep

    from plan2vec.dqn.point_mass_q_learning import Args, train
    from plan2vec_experiments import instr, config_charts

    jaynes.config("cpu")

    seeds = [s * 100 for s in range(5)]

    with Sweep(Args) as sweep:

        Args.num_episodes = 2000

        with sweep.product:
            with sweep.zip:
                Args.n_rollouts = [1, 3, 5, 10, 20, 30, 50, 100, 200, 300, 400, 500, 600, 900, 1800, 2400]
                Args.num_envs = [1, 1, 1, 1, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20, 20]

            Args.start_seed = seeds

    for kwargs in sweep:
        jaynes.run(instr(train, __postfix=f"n_rollouts-({Args.n_rollouts})", **kwargs))

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
