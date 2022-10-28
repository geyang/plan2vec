from plan2vec.plotting.maze_world.connect_the_dots_state_maze import Args, main

if __name__ == "__main__":
    import jaynes
    from plan2vec_experiments import instr

    # os.makedirs("figures", exist_ok=True)

    local_metric_exp_path = "episodeyang/plan2vec/2019/05-05/c-maze/c_maze_local_metric_sweep/13.29/01.187654"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"

    jaynes.config('vector-gpu')

    for Args.env_id in [
        "CMazeDiscreteIdLess-v0",
        "GoalMassDiscreteIdLess-v0",
        "回MazeDiscreteIdLess-v0"]:
        jaynes.run(instr(main, **vars(Args)))

    jaynes.listen()
    # main(**vars(Args))
