Fig. 1. 

Planning Success vs Planning Depth (Fan Out)

Fig. 2.

Planning Success vs Goal Distance

Fig. 3.

Plan2Vec Learning from Re-sampled DataSets

Does Sample Size Improve Performance? Keep the local-metric dataset size fixed.
Investigate


---
Old stuff

Biggest Benefit of a distilled Value Function is that it could contain transferable knowledge. However, Planning does better during exploration. Compare with naive exploration strategy, planning based exploration is more efficient at exploring the environment. Can compare under sparse reward. If the Planner explicitly involves a distilled Value function, then it should do much better than exploration at random.

----

Compare Plan2Vec with standard Q-learning -- no resampled

What is the benefit of planning?

1. planning works better when the state is complex, but the branching
   ratio is low. 

In comparison to RL, planning is able to generalize at test-time much better

In comparison to RL, planning is better as exploration.

In comparison to UVFA, planning generalize to non-trivial topological space better.
A neural network has trouble generalizing. <== is this even true?
