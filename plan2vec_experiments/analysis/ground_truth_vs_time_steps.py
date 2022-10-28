"""
Here we visualize the ground-truth distance vs the timesteps in a trajectory.

yAxis: Ground-truth distance
xAxis: Time-steps
2nd-yAxis: Variance of the distance
"""

import matplotlib.pyplot as plt

# sample trajectories

# autonomous manipulation in unstructured environments
# diverse, collaborative, and unstructured environments

from plan2vec.plotting.maze_world.state_maze_gt_vs_timesteps import Args, main

if __name__ == "__main__":
    Args.env_id = "GoalMassDiscreteIdLess-v0"
    main(**vars(Args))
