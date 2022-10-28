colors = []


def class_embedding_2d(name, samples_c, n_class=0, labels=None, title='Latent Embedding', ):
    """
    visualizes class clusters on a 2d domain. Color indicate different class labels.

    When no labels are passed-in, all points are plotted with the same color.

    :param samples_c:
    :param n_class:
    :param labels: tensor for the ground-truth labels
    :return:
    """
    import matplotlib.pyplot as plt
    from ml_logger import logger

    plt.figure(figsize=(3.5, 3), dpi=300)

    plt.title(title)
    for i in range(n_class):
        mask = labels == i
        plt.scatter(samples_c[:, 0][mask], samples_c[:, 1][mask], label=f"{i}", alpha=0.8, linewidth=0)

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.legend(loc="upper left", bbox_to_anchor=(0.95, 1), framealpha=1, frameon=False, fontsize=8)
    logger.savefig(f"figures/{name}.png", dpi=300, box_inches="tight")
    plt.close()
    # plt.show()
