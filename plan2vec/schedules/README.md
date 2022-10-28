# Schedulers

- [ ] Need to add documentation

### Enhancement Plans

@yayitsamyzhang If we need features on this let me know.

## Dilated Cosine Scheduler

> Note that the first argument is `max`, because the mininum learning rate is usually set 0.

![./figures/CosineAnneal(min=0.04,%20max=0.1,%20n=1500,%20k=4).png](./figures/CosineAnneal(min=0.04,%20max=0.1,%20n=1500,%20k=4).png)

The example code is below
```python
if __name__ == "__main__":
    s = CosineAnneal(0.1, min=0.04, n=1500, k=4)
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 2), dpi=300)
    plt.title(f'{s}')
    plt.plot([s.send(x) for x in range(1500)])
    plt.ylim(-0.1, 0.2)
    plt.savefig(f"figures/{s}.png")
    # plt.show()
```

## Dilated Delta Annearler

![./figures/DeltaAnneal(min=0.04,%20max=0.1,%20n=1500,%20k=4).png](./figures/DeltaAnneal(min=0.04,%20max=0.1,%20n=1500,%20k=4).png)


The example code is below
```python
if __name__ == "__main__":
    s = DeltaAnneal(0.1, min=0.04, n=1500, k=4)
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 2), dpi=300)
    plt.title(f'{s}')
    plt.plot([s.send(x) for x in range(1500)])
    plt.ylim(-0.1, 0.2)
    plt.savefig(f"figures/{s}.png")
    # plt.show()
```
