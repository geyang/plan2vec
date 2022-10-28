import os
import re

from matplotlib import rcParams
from termcolor import cprint

from plan2vec_experiments import RUN
from ml_logger import logger
import pandas as pd
from pprint import pprint


class ExpRuns:
    vanilla = "amyzhang/plan2vec-experiments/2019/02-04/amy_remote_debug/13.49.39/GoalMassDiscreteIdLess-v0-vanilla-pr597576"
    root = "amyzhang/plan2vec-experiments/2019/02-04"


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    logger.configure(RUN.server, ExpRuns.root, register_experiment=False)

    colors = ['#49b8ff', '#ff7575', '#66c56c', '#f4b247']
    line_styles = ['-', '-.', '--', ':']

    rcParams['axes.titlepad'] = 15
    rcParams['axes.labelpad'] = 12
    rcParams['axes.labelsize'] = 13

    fig = plt.figure(figsize=(6, 3), dpi=300)

    for q_fn in ['vanilla', 'shared-encoder', 'l2-embed-T', 'l2-embed']:
        query = re.compile(f'new_q.*GoalMassDiscreteIdLess-v0.*{q_fn}-seed')

        experiments = [_ for _ in logger.glob('**/metrics.pkl') if query.search(_)]
        if not experiments:
            continue
        cprint(experiments, 'green')

        df = pd.DataFrame(data=sum([logger.load_pkl(p) for p in experiments], []))

        c = colors.pop(0)
        line_style = line_styles.pop(0)
        data = df['success_rate/mean'].rolling(20, axis=0)
        plt.plot(df['episode'], data.quantile(.5, interpolation='linear'), color=c, linestyle=line_style, label=q_fn)
        a = plt.fill_between(df['episode'],
                             data.quantile(0.25, interpolation='linear'),
                             data.quantile(0.75, interpolation='linear'),
                             color=c, alpha=0.3, linewidth=0, label="")

    plt.ylim(-0.05, 1.05)

    plt.title('Learning Curves')
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1.0), framealpha=1, frameon=False, fontsize=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    logger.configure(prefix='', register_experiment=False)
    logger.remove('./figures/Learning_Curves.png')
    logger.savefig('./figures/Learning_Curves.png', bbox_inches='tight')
    # plt.show()
