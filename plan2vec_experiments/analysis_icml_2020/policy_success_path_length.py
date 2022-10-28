import pandas as pd
import numpy as np

from plan2vec_experiments.analysis_icml_2020 import plot_line, stylize

if __name__ == '__main__':
    df = pd.DataFrame(dict(
        index=np.arange(200)[::-1] / 15,
        ours=0.27 * (np.arange(200) ** .2 + np.random.rand(200)),
        dQN=0.08 * (np.arange(200) ** .3 + np.random.rand(200)),
        SPTM=0.02 * (np.arange(200) ** 0.5 + np.random.rand(200)),
        VAE=0.1 + 0.008 * (np.arange(200) ** 0.5 + np.random.rand(200)),
    ))

    stylize()
    plot_line(df, "index", 'ours', 'dQN', 'SPTM', 'VAE',
              title="Policy Success vs Distance-to-Goal",
              figsize=(4.2, 2.5),
              xlabel="Path Length", ylabel="Success Rate",
              ylim=(0, 1),
              ypc=True,
              filename="figures/policy_success_path_length.png")
