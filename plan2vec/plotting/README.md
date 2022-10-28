# Visualization Utilities for GMO

## Visualizing Embedding

To plot the embedding that you learned, use the `gmo.plotting.embedding_2D:visualize_embedding_2d` function.

```python
import matplotlib.pyplot as plt
from gmo.plotting.embedding_2D import visualize_embedding_2d, get_ball_fn


fig = plt.figure(figsize=(3.2, 3))
_ = get_ball_fn()
visualize_embedding_2d(title=f'Point Mass, {_.__name__}', embed_fn=_)
plt.close()
```

And it will generate some images as the following.

<p align="center">
    <img src="./figures/Point%20Mass%20(State).png" width="25%">
    <img src="./figures/Point%20Mass,%20ball(-0.10).png" width="25%">
    <img src="./figures/Point%20Mass,%20ball(0.10).png" width="25%">
</p>

```python
:param title:    the title of your plot. If no figure filename is specified, this is used as the figure name.
:param embed_fn: the embedding function, typically your neural network.
:param low:      the lower end of the range of x, y
:param high:     the higher end of the range of x, y
:param n:        number of bins for x, and y. should always be odd.
:param filename: Default to f"figures/{title}.png". If you want to overwrite this make
                 sure you use some ./figures/ namespace.
    
:return: None
```

## Visualizing Q-values

To plot the Q-value function, use the `gmo.plotting.plot_q_value:visualize_q_2d` function.

```python
from gmo.plotting.plot_q_value import visualize_q_2d

# An example Q-function
l2_Q = lambda xys, goal_xys: np.tile(- np.linalg.norm(goal_xys - xys, ord=2, axis=-1)[:, None], 9)

logger.configure(register_experiment=False)
visualize_q_2d(q_fn=l2_Q, cmap='RdBu', title='Cartesian Distance', key='figures/Cartesian Distance.png')
```

And it would save the plot as 

<p align="center">
<img src="./figures/Cartesian Distance.png" alt="Q-value function visualization">
</p>


