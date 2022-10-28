# Plan2Vec on DeepMind Streetlearn

## How to debug?

Getting supervised learning to work is a lot faster. We need to make sure that the network can actually over-fit.

1. 2D visualization: show color array for the map
2. how about supervised coordinates?
1. coordinate supervision 
2. value function supervision + different set of parameters
3. value iteration w/ ground-truth neighbors
4. value iteration w/ local function
1. review models, see which ones are needed for streetlearn

Might want to compare embedding learned via local metric function vs those of global.

## Done

- [x] oracle planner baseline: [73±5% on small and medium]
- [x] choose a different model / add a collection of models


## Scaling Factors for the various datasets

- Tiny: [524.10825964 697.14084164]
- Small: [251.83783118 286.49520676]
- Medium: [154.28508823 210.69826202]
- Large: [100.0057638  125.21619651]
- xl: [50.01646301 62.51763628]
