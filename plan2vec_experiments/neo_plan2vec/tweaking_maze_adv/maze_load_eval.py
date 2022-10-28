"""
Past Experiments

# env_id = "GoalMassDiscreteImgIdLess-v0"

# checkpoint = f"/geyang/plan2vec/2020/01-25/neo_plan2vec/tweaking_maze_adv/maze_env_sweep/17.31/10.896388/models/1800/adv.pkl"
# checkpoint = f"/geyang/plan2vec/2020/01-25/neo_plan2vec/tweaking_maze_adv/c-maze_tweak/own-trunk/12.46/adv_lr-1e-05-bp_scale-0.1-nun_rollouts-2000/28.563095/models/{1100:04d}/adv.pkl"
# checkpoint = f"/geyang/plan2vec/2020/01-25/neo_plan2vec/tweaking_maze_adv/c-maze_tweak/own-trunk/12.17/adv_lr-1e-05-bp_scale-0-nun_rollouts-5000/22.616392/models/{1000:04d}/Φ.pkl"
# checkpoint = f"/geyang/plan2vec/2020/01-25/neo_plan2vec/tweaking_maze_adv/c-maze_tweak/own-trunk/12.17/adv_lr-1e-05-bp_scale-0-nun_rollouts-5000/20.195152/models/{900:04d}/adv.pkl"
# checkpoint = f"/geyang/plan2vec/2020/01-25/neo_plan2vec/tweaking_maze_adv/c-maze_tweak/own-trunk/11.22/adv_lr-1e-05-bp_scale-0-nun_rollouts-1600/21.498162/models/{1200:04d}/adv.pkl"
# checkpoint = f"/geyang/plan2vec/2020/01-25/neo_plan2vec/tweaking_maze_adv/c-maze_tweak/own-trunk/11.22/adv_lr-1e-05-bp_scale-0-nun_rollouts-1600/26.705066/models/{1300:04d}/adv.pkl"
# checkpoint = f"/geyang/plan2vec/2020/01-25/neo_plan2vec/tweaking_maze_adv/c-maze_tweak/own-trunk/11.22/adv_lr-1e-05-bp_scale-0.3-nun_rollouts-1600/22.324716/models/{1200:04d}/adv.pkl"

"""

from plan2vec.plan2vec.maze_plan2vec import Args, eval_advantage

if __name__ == '__main__':
    from plan2vec_experiments import instr

    Args.latent_dim = 10
    Args.env_id = "CMazeDiscreteImgIdLess-v0"
    checkpoint = f"/amyzhang/plan2vec/2020/01-28/neo_plan2vec/tweaking_maze_adv/maze_load_adv_tweak/longer/2xv_s_s/07.44/CMazeDiscreteImgIdLess-v0/ams-0.1/hard/32.244512/models/{3800:04d}/adv.pkl"

    thunk = instr(eval_advantage, Args.env_id, checkpoint)
    thunk()
    exit()

# #%%
#
# # Lets try to plot the correlation
#
# import matplotlib.pyplot as plt
# from ml_logger import logger
# import numpy as np
#
# data = logger.summary_cache.data
# plt.figure(figsize=(10, 3), dpi=200)
# # plt.plot(data['success'], color='#23aaff')
# plt.plot(data['reset'], color='blue', label="reset")
# plt.gca().twinx()
# plt.plot(data['goal_pos'], color='red', linewidth=3, alpha=0.3, label="goal position")
# plt.plot(data['pos'], color='green', label="goal position")
# plt.plot(data['success'], color='orange', label="success")
# plt.legend(frameon=False)
# plt.tight_layout()
# logger.savefig(f"figures/cmaze_trace_3.png")
# plt.show()
#
# plt.plot(data['success'], color='#23aaff')
# plt.gca().twinx()
# plt.plot(data['d2goal'], color='red', alpha=0.3)
# plt.show()
