import pandas as pd
import numpy as np

from plan2vec_experiments.analysis_icml_2020 import plot_line, stylize

if __name__ == '__main__':
    df = pd.DataFrame(dict(
        index=np.arange(200),
        ours=np.concatenate([
            np.ones(50),
            np.min([np.ones(100), 0.005 * np.arange(100) ** 2 + np.ones(100) * 0.8], axis=0),
            np.min([np.ones(50), 0.005 * np.arange(50) ** 2 + np.ones(50) * 0.8], axis=0),
        ]) - 0.05 * np.random.rand(200),
        dQN=np.concatenate([
            np.ones(50),
            0.00001 * np.arange(100) ** 2 + np.ones(100) * 0.14,
            0.00001 * np.arange(50) ** 2 + np.ones(50) * 0.14,
        ]) - 0.05 * np.random.rand(200),
        SPTM=np.concatenate([np.ones(50), np.ones(150) * 0.14]) - 0.05 * np.random.rand(200),
        VAE=np.ones(200) * 0.14 + 0.05 * np.random.rand(200),
    ))

    stylize()
    plot_line(df, "index", 'ours', 'dQN', 'SPTM',  # 'VAE',
              title="Policy Success in A Changing Environment",
              figsize=(4.2, 2.5),
              xlabel="Training Epochs", ylabel="Success Rate",
              ylim=(0, 1),
              ypc=True,
              filename="figures/adaptation_success.png")

if __name__ == '__main__':
    df = pd.DataFrame(dict(
        index=np.arange(200),
        ours=140 + 600 * (1 - np.concatenate([
            np.ones(50),
            np.min([np.ones(100), 0.005 * np.arange(100) ** 2 + np.ones(100) * 0.8], axis=0),
            np.min([np.ones(50), 0.005 * np.arange(50) ** 2 + np.ones(50) * 0.8], axis=0),
        ]) - 0.05 * np.random.rand(200)),
        SPTM=1400 * np.ones(200),
    ))

    stylize()
    plot_line(df, "index", 'ours', 'SPTM',  # 'dQN',  # 'VAE',
              title="Planning Cost During Adaptation",
              figsize=(4.2, 2.5),
              xlabel="Training Epochs", ylabel="Planning Cost",
              ylim=(0, None),
              # color='grey',
              filename="figures/adaptation_cost.png")
