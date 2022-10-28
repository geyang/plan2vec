if __name__ == '__main__':
    import jaynes
    from params_proto.neo_hyper import Sweep

    from plan2vec.dqn.point_mass_q_learning import Args, train
    from plan2vec_experiments import instr, config_charts

    SEEDS = [s * 100 for s in range(10, 15)]

    jaynes.config()

    with Sweep(Args) as sweep:

        Args.num_episodes = 2000
        Args.env_id = 'GoalMassDiscreteImgIdLess-v0'
        # note: this environment uses a fixed goal
        # Args.env_id = 'GoalMassDiscreteFixGImgIdLess-v0'
        Args.num_envs = 10
        Args.batch_timesteps = 20
        Args.obs_key = 'img'
        Args.goal_key = 'goal_img'
        Args.q_fn = "ResNet18CoordQ"
        Args.latent_dim = 2
        Args.metric_summary_interval = 1
        Args.lr = 1e-4  # for image domain, this is good. 3e-5 is slower.
        Args.batch_size = 16
        Args.optim_steps = 15
        Args.prioritized_replay = True
        Args.target_update = 10
        Args.replay_memory = 500
        # Args.good_relabel = True

        Args.checkpoint_interval = 200

        with sweep.product:
            Args.start_seed = SEEDS

        # with sweep.zip:
        #     Args.n_rollouts = [10, 20, 30, 50, 100, 200, 300, 500, 800, ]
        #     Args.num_envs = [1, 10, 10, 10, 20, 20, 20, 20, 20, ]

    for deps in sweep:
        jaynes.run(
            instr(train, deps, __postfix=f"{Args.env_id}-{Args.q_fn}"))

        config_charts("""
        charts:
          - yKey: q_loss/mean
            xKey: episode
          - yKey: success_rate/mean
            xKey: episode
            yDomain: [-0.1, 1]
          - yKey: eval/success_rate/mean
            xKey: episode
            yDomain: [-0.1, 1]
          - {glob: '**/*.mp4', type: video}
        keys:
          - run.status
          - Args.n_rollouts
          - Args.num_envs
          - Args.lr
          - Args.eps_greedy
          - DEBUG.ground_truth_neighbor_r
          - DEBUG.supervised_value_fn
          - Args.term_r
        """)

    jaynes.listen()
