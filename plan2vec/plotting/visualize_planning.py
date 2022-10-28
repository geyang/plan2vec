def visualize_one_step_plans(x, xg, ns, x_star, key=None):
    from ml_logger import logger
    import matplotlib.pyplot as plt

    plt.figure(figsize=(3, 3), dpi=300)
    plt.title('1-Step Plans (Ground Truth)')

    plt.scatter(x[0], x[1], c="#e6e6e6")
    plt.scatter(ns[:, 0], ns[:, 1], c="green", alpha=0.1, edgecolors='none')
    plt.plot([x[0], x_star[0]], [x[1], x_star[1]], '-', color='gray')
    plt.scatter(x_star[0], x_star[1], c="#23aaff", edgecolors='none')
    plt.plot([x[0], xg[0]], [x[1], xg[1]], '-', color='lightgray')
    plt.scatter(xg[0], xg[1], c="red", edgecolors='none')

    plt.gca().set_aspect('equal')
    plt.ylim(-0.3, 0.3)
    plt.xlim(-0.3, 0.3)

    if key is None:
        plt.show()
    else:
        logger.savefig(key)
    plt.close()


def visualize_rope_plans(path, src, dest, all_images, traj_labels, traj_starts, _ds):
    from math import ceil
    import matplotlib.pyplot as plt
    from ml_logger import logger

    n_rows = ceil(len(path) / 10)
    fig = plt.figure(figsize=np.array(1.4, dtype=int) * [10, n_rows])
    plt.suptitle(f'Summary of Plan ({src}, {dest})', fontsize=14)
    gs = plt.GridSpec(n_rows, 10)

    _row = 0
    _col = 0
    for idx in range(len(path)):
        if _col >= 10:
            _row += 1
            _col = 0
        plt.subplot(gs[_row, _col])
        plt.imshow(all_images[path[idx]].squeeze(0), cmap='gray')
        traj_ind = traj_labels[path[idx]]
        img_ind = path[idx] - traj_starts[traj_ind]
        plt.text(0, 10, f"traj:{traj_ind}:{img_ind}", fontsize=8)
        if idx > 0:
            score = _ds[path[idx - 1], path[idx]]
            plt.text(0, 60, f"{score:0.2f}", fontsize=8, color="red" if score > 1.1 else "black")
        plt.axis('off')
        _col += 1
    plt.show()