"""
Sweeping over a few different maze environments, to
collect baseline measurements on how well the advantage learning works.

"""
from plan2vec.plan2vec.maze_plan2vec import Args, train
from plan2vec_experiments import instr

c_maze_prefix = \
    "/geyang/plan2vec/2020/01-25/neo_plan2vec/tweaking_maze_adv/maze_env_sweep/18.57"
c_maze_weights = ['14.810421/models/0800/Φ.pkl', '09.741905/models/0800/Φ.pkl',
                  '13.581646/models/0800/Φ.pkl', '14.098215/models/0800/Φ.pkl',
                  '13.067490/models/0800/Φ.pkl']

goal_mass_prefix = \
    "/geyang/plan2vec/2020/01-25/neo_plan2vec/tweaking_maze_adv/maze_env_sweep/17.31"
goal_mass_weights = ['12.499480/models/0800/Φ.pkl', '09.505578/models/0800/Φ.pkl',
                     '14.130932/models/0800/Φ.pkl', '10.896388/models/0800/Φ.pkl',
                     '03.890234/models/0800/Φ.pkl']

LOAD_PATHS = [
    *[goal_mass_prefix + "/" + p for p in goal_mass_weights],
    *[c_maze_prefix + "/" + p for p in c_maze_weights],
]

# def get_pretrained(prefix):
#     from ml_logger import logger
#     with logger.PrefixContext(prefix):
#         weights = logger.glob("**/models/0800/Φ.pkl")
#     return weights
#
#
# if __name__ == '__main__':
#     weights = instr(get_pretrained)(c_maze_prefix)
#     print(weights)
#     weights = instr(get_pretrained)(goal_mass_prefix)
#     print(weights)
#
#     exit()

if __name__ == '__main__':
    import jaynes
    from params_proto.neo_hyper import Sweep
    from plan2vec_experiments import instr, config_charts
    from ml_logger import logger

    jaynes.config()

    with Sweep(Args) as sweep:
        Args.num_epochs = 4000
        Args.limit = 2
        Args.num_rollouts = 600
        Args.checkpoint_interval = 200

        Args.start_epoch = 801
        Args.latent_dim = 10
        Args.eval_soft = False
        with sweep.zip:
            Args.seed = range(5)
            Args.load_global_metric = LOAD_PATHS[-5:]

    for deps in sweep:
        thunk = instr(train, deps, __prefix='learn-adv',
                      __postfix=f"{Args.env_id}/ams-{Args.adv_mean_scale}{'-hard' if Args.eval_soft else ''}")
        jaynes.run(thunk, )
        config_charts("""
            charts:
            - yKey: adv_act/mean
              xKey: epoch
              yDomain: [-0.05, 0.05]
            - yKey: adv_target/mean
              xKey: epoch
              yDomain: [-0.05, 0.05]
            - yKey: adv_values/mean
              xKey: epoch
              yDomain: [-0.05, 0.05]

            - yKey: adv_loss/mean
              xKey: epoch
              yDomain: [0, 0.0035]

            - yKey: success/mean
              xKey: epoch
            keys:
            - run.status
            - run.job_id
            - Args.num_rollouts
            - Args.adv_bp_scale
            - Args.adv_bp_tweak
            """)
        logger.log_text(__doc__, "README.md")

    jaynes.listen(600)
