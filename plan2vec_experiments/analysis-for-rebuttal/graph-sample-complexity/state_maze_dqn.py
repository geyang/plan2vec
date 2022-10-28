if __name__ == '__main__':
    import jaynes
    from params_proto.neo_hyper import Sweep

    from plan2vec.dqn.point_mass_q_learning import Args, train
    from plan2vec_experiments import instr, config_charts

    jaynes.config("vector")

    with Sweep(Args) as sweep:

        Args.num_episodes = 4000

        with sweep.zip:
            Args.env_id = [
                "GoalMassDiscreteIdLess-v0",
                "回MazeDiscreteIdLess-v0",
                "CMazeDiscreteIdLess-v0",
            ]
            Args.start_seed = [*range(3)]

    for kwargs in sweep:
        jaynes.run(instr(train, __postfix=f"env({Args.env_id})/seed({Args.start_seed})", **kwargs))

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
