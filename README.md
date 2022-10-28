# Plan2Vec: Unsupervised Representation Learning by Latent Plans 

**Official repo for [Plan2Vec](https://geyang.github.io/plan2vec).**

We are working on a clean implementation of the expert distillation variant of the plan2vec algorithm for public release.

<p align="center"><img alt="Overview of Plan2vec" src="figures/plan2vec_main.png" width="788" height="175"/></p>

## Code

This code base has been split up into a few different components that each lives in its own repository.
- For planning, code, refer to the `graph-search` package on PyPI (https://github.com/geyang/graph-search). This is my canonical implementation of the graph planning algorithms and is used in a number of other projects. This repo contains visualization of planing results. 
- Here is the pre-processing pipeline for the StreetLearn dataset. I reverse engineered the buffer format: https://github.com/geyang/streetlearn
- The `plan2vec` module should be inside [./plan2vec](./plan2vec). The best result was given under the supervised mode, where the distance between samples are given by the shortest path between the corresponding nodes on the graph.

All experiment scripts live in the `plan2vec_experiments` folder.

## BibTex

```
@inproceedings{yang2020plan2vec,
    title={Plan2vec: Unsupervised Representation Learning by Latent Plans},
    author={Yang, Ge and Zhang, Amy and Morcos, Ari S. and Pineau, Joelle and Abbeel, Pieter and Calandra, Roberto},
    booktitle={Proceedings of The 2nd Annual Conference on Learning for Dynamics and Control},
    series={Proceedings of Machine Learning Research},
    pages={1-12},
    year={2020},
    volume={120},
    note={arXiv:2005.03648}
}
```

