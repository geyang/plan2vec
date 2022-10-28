import numpy as np
from params_proto.neo_proto import ParamsProto
from functools import lru_cache


class Args(ParamsProto):
    gamma = 0.9
    lam = 0.9

    T = 20  # also use as H, truncated TD(λ)


# How shall we use this?
@lru_cache()
def td_lambda(T, lam, gamma):
    # assume that last state is the terminal state.
    el_rewards = np.zeros(T)
    el_states = np.zeros(T)
    # We fix the G_t to the left side, and focus
    # on computing the target value on the right
    # hand side.
    for n in range(1, T - 1):
        # 1 step:
        for t in range(0, n):
            el_rewards[t] += gamma ** t * lam ** (n - 1)
        el_states[n] += gamma ** n * lam ** (n - 1)
    return el_rewards, el_states


def td_target(next_states, rewards, goal, value_fn, lam, gamma):
    assert len(next_states) == len(rewards), "input [t+1:T] states"
    el_r, el_s = td_lambda(len(next_states) + 1, lam, gamma)
    value_target = el_r[1:] * rewards + el_s[1:] * value_fn(next_states, goal[None, ...])
    # add normalization
    return value_target.sum() / el_s[1:].sum()


def mc_target(rewards):
    return rewards.sum(-1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    colors = ['#49b8ff', '#66c56c', '#f4b247', '#ff7575']
    xticks = [1, 2, 3, 4, 5, 10, 15, 20]

    plt.figure(figsize=(3, 2), dpi=200)
    plt.title('TD-λ (Unnormalized)')
    for i, Args.lam in enumerate([0.99, 0.9, 0.5, 0]):
        _, el = td_lambda(Args.T, Args.lam, Args.gamma)
        plt.plot(range(1, len(el)), el[1:], 'o-', markeredgecolor='white', markersize=4,
                 color=colors[i % 4], alpha=0.8, label=f'{Args.lam}')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().xaxis.set_ticks(xticks)
    plt.ylim(None, 1.1)
    plt.xlim(None, 20)
    plt.legend(frameon=False)
    plt.show()

    plt.figure(figsize=(3, 2), dpi=200)
    plt.title('TD-λ (Normalized)')
    for i, Args.lam in enumerate([0.99, 0.9, 0.5, 0]):
        _, el = td_lambda(Args.T, Args.lam, Args.gamma)
        plt.plot(range(1, len(el)), el[1:] / el[1:].sum(), 'o-', markeredgecolor='white',
                 markersize=4, color=colors[i % 4], alpha=0.8, label=f'{Args.lam}')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().xaxis.set_ticks(xticks)
    plt.ylim(None, 1.1)
    plt.xlim(None, 20)
    plt.legend(frameon=False)
    plt.show()
