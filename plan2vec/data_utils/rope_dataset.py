from bisect import bisect
from copy import copy

import numpy as np
from torch.utils.data import Dataset, DataLoader


class PairDataset(Dataset):
    """ Pair Dataset: supports fractional indices for k-fold validation. """

    def __init__(self, data, shuffle=True):
        """ Creates a Paired dataset object

        :param data: numpy list of trajectory tensors
        :param shuffle: boolean flag for whether shuffle the order of different trajectories
        """
        if shuffle:  # shuffles by trajectory.
            data = copy(data)
            np.random.shuffle(data)

        x, x_prime = [], []
        for traj in data:  # Iterate through each trajectory
            traj = traj.astype(np.float32)
            for j in range(len(traj) - 1):
                x.append(traj[j].transpose(2, 0, 1) / 255)
                x_prime.append(traj[j + 1].transpose(2, 0, 1) / 255)

        self.data = list(zip(np.concatenate([x, x]),
                             np.concatenate([x_prime, np.random.permutation(x_prime)])))
        self.labels = np.concatenate([np.ones(len(x), dtype=np.float32),
                                      np.zeros(len(x_prime), dtype=np.float32)])

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f'PairDataset("")'


# support for fractional indexing
# if isinstance(index, slice):
#     index = slice(*[int(i * len(self.data)) if isinstance(i, float) else i
#                     for i in (index.start, index.stop, index.step)])
#     _ = copy(self)
#     _.data = self.data[index]
#     _.labels = self.labels[index]
#     return _

class SeqDataset(Dataset):
    def __init__(self, data, H=20, shuffle=None):
        self.H = H
        if shuffle:
            data = copy(data)
            np.random.shuffle(data)

        self._cum_size = [0]
        for i in range(len(data)):  # Iterate through each trajectory
            self._cum_size += [self._cum_size[-1] + len(data[i]) - H]

        self.data = data

    def __getitem__(self, index):
        file_index = bisect(self._cum_size, index) - 1
        seq_index = index - self._cum_size[file_index]
        data = self.data[file_index][seq_index:seq_index + self.H]
        return data.astype(np.float32).transpose(0, 3, 1, 2) / 255

    def __len__(self):
        return self._cum_size[-1]

    def __repr__(self):
        return f'SequenceDataset("")'


if __name__ == "__main__":
    # write the tests here
    from os.path import expanduser

    data_path = "~/fair/new_rope_dataset/data/new_rope.npy"
    rope = np.load(expanduser(data_path))
    seq_loader = DataLoader(SeqDataset(rope, 20, True), batch_size=30, shuffle=True, num_workers=1)
    seq_gen = iter(seq_loader)
    for i in range(1000):
        paths = next(seq_gen)
        print(paths.shape)
