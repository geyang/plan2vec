if __name__ == '__main__':
    import jaynes

    from plan2vec.dqn.point_mass_q_learning import Args, train
    from plan2vec_experiments import instr, config_charts
    from params_proto.neo_hyper import Sweep

    jaynes.config('vector-cpu')

    with Sweep(Args) as sweep:
        with sweep.product:
            Args.eps_greedy = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0]

    for deps in sweep:
        jaynes.run(instr(train,
                         deps,
                         __postfix=f"eps-greedy({Args.eps_greedy})", ))
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
    jaynes.listen()
