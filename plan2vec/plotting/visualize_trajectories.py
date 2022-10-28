from textwrap import dedent

from params_proto import cli_parse


@cli_parse
class Args:
    # exp_dir = "/geyang/plan2vec/2019/06-24/c-maze-image/plan2vec_tweak/gt_neighbor_3d/10.14/35.349777"
    # exp_dir = "/geyang/plan2vec/2019/07-02/c-maze-image/plan2vec/plan2vec_3d-1-step/dense/GlobalMetricCoordConvL2/lr-(0.003)/22.53/32.825842"
    exp_dir = "/geyang/plan2vec/2019/07-02/c-maze-image/plan2vec/plan2vec_3d/dense/ResNet18L2/lr-(0.001)/21.55/45.574720"


def visualize_trajectories():
    from ml_logger import logger
    import matplotlib.pyplot as plt

    with logger.PrefixContext(Args.exp_dir):
        paths = logger.glob("plans/*.pkl")

        paths = sorted(paths)

        print(*paths, sep='\n')

    for i, p in enumerate(paths):
        with logger.PrefixContext(Args.exp_dir):
            data, = logger.load_pkl(p)

        s = data['s']
        s_goal = data['s_goal']

        import numpy as np
        plt.figure(figsize=(3, 3), dpi=70)
        for xys, goals in zip(s.transpose(1, 0, 2), s_goal.transpose(1, 0, 2)):
            plt.scatter(xys[:, 0], xys[:, 1], c=np.arange(len(xys[:, 0])), linewidth=4, s=20, alpha=1)
            plt.scatter(goals[:, 0], goals[:, 1], c=np.arange(len(goals[:, 0])), marker="x", linewidth=4, s=60, alpha=1)
            break

        plt.xlim(-0.3, 0.3)
        plt.ylim(-0.3, 0.3)
        logger.savefig(p.replace('pkl', 'png'))
        logger.savefig(p.replace('pkl', 'pdf'))
        plt.close()

    print('done')


if __name__ == "__main__":
    from plan2vec_experiments import instr, config_charts
    import jaynes

    jaynes.config('local')
    _ = instr(visualize_trajectories)
    config_charts("""
    charts:
    - type: file
      glob: "**/*.png"
    """)
    jaynes.run(_)
    jaynes.listen()
