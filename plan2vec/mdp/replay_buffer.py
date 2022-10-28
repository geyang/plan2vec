import numpy as np
import random
from collections import namedtuple, deque, defaultdict

from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree


class ArrayDict(dict):
    def __init__(self, d=None, **kwargs):
        super().__init__()
        kwargs.update(d or {})
        for k, v in kwargs.items():
            self[k] = np.array(v)

    def __getitem__(self, item):
        if type(item) is str:
            return dict.__getitem__(self, item)

        _ = {}
        for k, v in self.items():
            v_ = np.array(v)
            inds_ = item if type(item) is tuple else np.broadcast_to(item, v_.shape)
            _[k] = v_[inds_]
        return ArrayDict(_)


class ReplayBuffer:
    __len = 0

    def __init__(self, capacity):
        self.buffer = defaultdict(lambda: deque(maxlen=capacity))

    def add(self, **kwargs):
        self.extend(**{k: [v] for k, v in kwargs.items()})

    def extend(self, **kwargs):
        """Saves a transition."""
        # TODO: add assert to check lens match
        for k, v in kwargs.items():
            self.buffer[k].extend(v)
        self.__len = len(self.buffer[k])

    def sample(self, batch_size: int, **__):
        """
        if batch_size larger than buffer length, returns buffer length

        :param batch_size: int, size for the sample batch.
        :return: dict
        """
        # need to change this to a generator. sample without replacement.
        batch = {}
        inds = np.random.rand(len(self)).argsort()[:batch_size]
        for i, (k, v) in enumerate(self.buffer.items()):
            _ = np.take(v, inds, axis=0)  # have to specify axis, otherwise flattens array.
            try:
                batch[k] = np.stack(_, axis=0)
            except ValueError:  # usually happens when the values are different shape.
                batch[k] = _
        return batch

    def __len__(self):
        return self.__len

    def __getitem__(self, inds):
        return ArrayDict(self.buffer)[inds]

    def clear(self):
        self.buffer.clear()


if __name__ == "__main__":
    # code for testing the ArrayDict class
    b = ReplayBuffer(5)
    b.extend(s=np.arange(20).reshape(-1, 2), a=np.arange(20).reshape(-1, 2))
    extended = b[None, :]
    print(extended)
    inds = b['s'] > 6
    print(inds)
    lt_six = extended[inds]
    print(lt_six)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super().__init__(size)
        # assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def extend(self, **kwargs):
        """See ReplayBuffer.store_effect"""
        super().extend(**kwargs)
        idx = len(self)
        num_samples = len(list(kwargs.values())[0])
        start_idx = idx - num_samples
        for i in range(start_idx, idx):
            self._it_sum[i] = self._max_priority ** self._alpha
            self._it_min[i] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        if beta == 0:
            return super().sample(batch_size)

        inds = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        batch = {}
        for idx in inds:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        batch['weights'] = weights
        batch['inds'] = inds
        for i, (k, v) in enumerate(self.buffer.items()):
            _ = np.take(v, inds, axis=0)  # have to specify axis, otherwise takes flattened array.
            batch[k] = np.stack(_, axis=0)
        return batch

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class EpisodicBuffer:
    __len = 0

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def extend(self, **kwargs):
        """Saves a transition."""
        # TODO: add assert to check lens match
        self.buffer.append(kwargs)
        self.__len = len(self.buffer)

    def sample(self, batch_size: int, **__):
        """
        if batch_size larger than buffer length, returns buffer length

        :param batch_size: int, size for the sample batch.
        :return: dict
        """
        # need to change this to a generator. sample without replacement.
        inds = np.random.rand(len(self)).argsort()[:batch_size]
        return self[inds]

    def __len__(self):
        return self.__len

    def __getitem__(self, inds):
        return [self.buffer[i] for i in inds]

    def clear(self):
        self.buffer.clear()


if __name__ == "__main__":
    # code for testing the ArrayDict class
    b = EpisodicBuffer(5)
    b.extend(s=np.arange(11).reshape(-1, 2), a=np.arange(11).reshape(-1, 2))
    print(b)
    traj = b.sample(1)
    traj
    # extended = b[None, :]
    # print(extended)
    # inds = b['s'] > 6
    # print(inds)
    # lt_six = extended[inds]
    # print(lt_six)
