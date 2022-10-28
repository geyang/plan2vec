from termcolor import cprint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
from os.path import expanduser
from ml_logger import logger
from heapq import heappush, heappop
import torch
from math import ceil


evals = [[ [17, 13, 5, 12, 16, 20, 24, 29, 33, 16, 20, 24, 28, 32, 34, 39, 39],
 [44, 40, 37, 40, 37, 40, 37, 40, 37, 40, 28, 18, 41, 16, 20, 24, 28],
 [16, 13, 10, 24, 28, 32, 41, 27, 28, 32, 41, 15, 19, 34, 32, 19, 17],
 [17, 21, 21, 24, 28, 32, 35, 9, 20, 24, 28, 32, 43, 34, 11, 4, 0],
 [38, 40, 37, 13, 5, 10, 7, 11, 15, 19, 24, 28, 32, 43, 6, 4, 0],
 [24, 25, 22, 18, 22, 10, 7, 11, 13, 5, 10, 7, 11, 13, 7, 13, 17],
 [43, 45, 44, 3, 7, 14, 17, 4, 28, 31, 34, 45, 21, 24, 28, 31, 32],
 [11, 12, 16, 20, 24, 28, 32, 43, 12, 16, 20, 24, 29, 33, 34, 44, 44],
 [18, 15, 11, 7, 11, 13, 22, 16, 12, 16, 20, 24, 29, 33, 34, 45, 45],
 [7, 12, 16, 20, 24, 28, 32, 43, 34, 32, 39, 20, 24, 28, 32, 35, 35],
 [48, 45, 43, 12, 16, 20, 24, 29, 33, 34, 45, 46, 42, 32, 28, 24, 24],
 [35, 31, 24, 16, 20, 24, 28, 32, 43, 12, 16, 20, 24, 28, 32, 34, 36],
 [20, 18, 17, 16, 17, 32, 30, 27, 15, 18, 15, 18, 21, 24, 28, 32, 43, 38],
 [25, 21, 18, 16, 20, 18, 16, 20, 18, 16, 20, 18, 16, 20, 18, 16, 20, 8],
 [0, 4, 7, 12, 16, 20, 24, 29, 33, 34, 12, 16, 20, 24, 28, 32, 43, 46],
 [9, 7, 11, 7, 11, 7, 11, 7, 11, 7, 11, 7, 22, 18, 17, 13, 5, 5],
 [43, 45, 44, 3, 7, 14, 17, 4, 28, 31, 34, 45, 21, 24, 28, 31, 23, 30],
 [9, 11, 15, 7, 12, 16, 20, 24, 29, 33, 43, 12, 21, 24, 28, 32, 43, 42],
 [11, 12, 16, 20, 24, 28, 32, 43, 12, 16, 20, 24, 29, 33, 34, 44, 28, 30],
 [4, 11, 12, 16, 19, 23, 17, 21, 12, 16, 20, 24, 28, 32, 43, 34, 45, 49],
 [49, 46, 42, 40, 37, 40, 37, 40, 37, 40, 37, 40, 37, 40, 37, 40, 37, 12],
 [17, 20, 24, 28, 32, 21, 24, 28, 32, 43, 12, 16, 20, 24, 29, 33, 34, 44],
 [4, 7, 12, 30, 27, 32, 34, 18, 15, 11, 5, 12, 16, 20, 24, 29, 33, 34],
 [48, 45, 43, 12, 16, 20, 24, 29, 33, 34, 45, 46, 42, 32, 28, 24, 31, 31],
 [35, 31, 5, 12, 16, 20, 24, 29, 33, 34, 44, 43, 18, 21, 24, 29, 33, 33],
 [7, 6, 44, 28, 32, 35, 18, 20, 24, 29, 33, 34, 45, 12, 16, 20, 24, 28],
 [3, 4, 43, 6, 12, 16, 20, 23, 5, 12, 16, 20, 24, 29, 33, 34, 45, 48],
 [41, 43, 34, 43, 34, 43, 34, 43, 34, 43, 34, 43, 34, 43, 34, 20, 24, 29],
 [1, 5, 11, 15, 19, 16, 20, 24, 28, 32, 41, 24, 28, 32, 34, 32, 30, 29],
 [35, 31, 24, 16, 20, 24, 28, 32, 43, 12, 16, 20, 24, 28, 32, 34, 25, 25],
 [3, 11, 12, 16, 19, 23, 27, 29, 24, 32, 31, 27, 3, 20, 24, 28, 32, 43, 47],
 [40, 44, 45, 44, 35, 29, 16, 20, 24, 29, 33, 34, 45, 25, 21, 19, 15, 11, 11],
 [25, 21, 18, 16, 20, 18, 16, 20, 18, 16, 20, 18, 16, 20, 18, 16, 20, 34, 34],
 [38, 40, 37, 13, 5, 10, 7, 11, 15, 19, 24, 28, 32, 43, 6, 4, 45, 43, 41],
 [21, 25, 28, 44, 26, 28, 32, 35, 12, 16, 20, 24, 28, 31, 24, 28, 32, 41, 39],
 [9, 7, 11, 7, 11, 7, 11, 7, 11, 7, 11, 7, 22, 18, 17, 13, 5, 28, 30],
 [32, 40, 37, 40, 37, 40, 37, 40, 37, 40, 37, 40, 37, 6, 12, 16, 20, 23, 30],
 [11, 12, 16, 20, 24, 28, 32, 43, 12, 16, 20, 24, 29, 33, 34, 44, 28, 28, 28],
 [46, 43, 10, 14, 15, 19, 23, 2, 24, 28, 32, 43, 26, 39, 35, 32, 28, 24, 24],
 [17, 16, 17, 16, 17, 16, 17, 16, 17, 16, 17, 16, 17, 45, 34, 3, 7, 13, 17],
 [45, 43, 36, 39, 33, 29, 44, 34, 44, 16, 20, 24, 28, 32, 41, 16, 20, 23, 30],
 [15, 18, 19, 18, 19, 18, 19, 18, 19, 18, 19, 18, 5, 5, 10, 7, 11, 13, 13],
 [13, 16, 20, 5, 28, 32, 34, 46, 42, 32, 28, 24, 17, 17, 10, 14, 15, 19, 22],
 [3, 4, 43, 6, 12, 16, 20, 23, 5, 12, 16, 20, 24, 29, 33, 34, 45, 30, 31],
 [8, 7, 13, 23, 27, 40, 42, 32, 28, 24, 24, 28, 32, 35, 23, 5, 11, 15, 18],
 [7, 12, 16, 20, 24, 23, 20, 17, 13, 7, 12, 16, 20, 24, 22, 19, 15, 11, 11],
 [33, 31, 27, 34, 16, 20, 24, 28, 32, 43, 16, 20, 24, 29, 33, 31, 34, 45, 49]]]

def build_edges(inds):
    edges = dict(enumerate(inds))
    return edges


def dijkstras(src, dest, edges):
    visited = set()
    unvisited = []
    distances = {}
    predecessors = {}

    distances[src] = 1
    heappush(unvisited, (1, src))

    while unvisited:
        # visit the neighbors
        dist, v = heappop(unvisited)
        if v in visited or v not in edges:
            continue
        visited.add(v)
        if v == dest:
            # We build the shortest path and display it
            path = []
            pred = v
            while pred != None:
                path.append(pred)
                pred = predecessors.get(pred, None)
            if len(path) == 1:
                print('foo', src == dest)
                print(src)
                print(dest)
                return None, None
            return path[::-1], 1 / dist

        neighbors = list(edges[v])

        for idx, neighbor in enumerate(neighbors):
            if neighbor not in visited:
                new_distance = distances[v]
                if new_distance < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_distance
                    heappush(unvisited, (new_distance, neighbor))
                    predecessors[neighbor] = v

    # couldn't find a path
    return None, None


def plot_traj(path, title, file_name, show_score=False):
    n_cols = len(path)
    fig = plt.figure(figsize=np.array(1.4, dtype=int) * [n_cols, 1])
    # plt.suptitle(title, fontsize=14)
    gs = GridSpec(1, n_cols)
    for idx in range(len(path)):
        plt.subplot(gs[0, idx])
        plt.imshow(all_images[path[idx]].squeeze(0), cmap='gray')
        traj_ind = traj_labels[path[idx]]
        img_ind = path[idx] - traj_starts[traj_ind]
        plt.text(0, 10, f"traj:{traj_ind}:{img_ind}", fontsize=8)
        if idx > 0 and show_score:
            score = _ds[path[idx - 1], path[idx]]
            plt.text(0, 60, f"{score:0.2f}", fontsize=8, color="red" if score > 1.1 else "black")
        plt.axis('off')

    logger.savefig(file_name, bbox_inches='tight')


_ds = np.load(file=expanduser("~/fair/pairwise.npy"))[:500, :500]

term_r = 1.3
with torch.no_grad():
    ds = torch.tensor(_ds)
    ds[torch.eye(ds.shape[0], dtype=torch.uint8)] = float('inf')
    for k in range(1, 3):  # add true neighbors
        diag = torch.diagflat(torch.ones(len(ds) - k, dtype=torch.uint8), k)
        ds[diag] = 0.5
        diag = torch.diagflat(torch.ones(len(ds) - k, dtype=torch.uint8), -k)
        ds[diag] = 0.5
    # top_cols, top_col_inds = torch.topk(_ds, k=k, dim=1, largest=False, sorted=True)
    full_range = torch.arange(len(ds))
    inds = [None] * len(ds)
    for idx, row in enumerate(ds):
        visited = []
        inds[idx] = full_range[row <= term_r].numpy()
        if len(inds[idx]) == 0:
            raise Exception('term_r too small')
            break
        for i in range(1):
            new_neighbors = [full_range[ds[id] <= term_r].numpy() for id in inds[idx] if id not in visited]
            if len(new_neighbors) > 0:
                neighbors = np.concatenate(new_neighbors)
            else:
                neighbors = []
            visited += list(inds[idx])
            inds[idx] = np.unique(np.concatenate([inds[idx], neighbors]))
            inds[idx] = inds[idx][inds[idx] != idx]  # remove identity
    inds = np.array(inds)
    top_ds = np.array([row[_] for row, _ in zip(ds.numpy(), inds)])


rope = np.load(expanduser("~/fair/new_rope_dataset/data/new_rope.npy"))[16:17]
all_images = np.concatenate(rope).transpose(0, 3, 1, 2)[:500]
traj_labels = np.concatenate([np.ones(len(traj)) * i for i, traj in enumerate(rope)]).astype(int)
traj_starts = [0]
for i in range(len(rope)):
    traj_starts.append(traj_starts[-1] + rope[i].shape[0])
assert len(all_images) == len(ds), "the dimension of the distance matrix and the images should agree."


edges = build_edges(inds)

for path in evals[0]:
    if path[-1] == path[-2]:
        path.pop(-1)

    src = path[0]
    dest = path[-1]
    try:
        plot_traj(path, f'Value Iteration Trajectory ({src}, {dest})', f'./figures/value_iter_{src}_{dest}.png', show_score=True)
        traj_ind = traj_labels[src]
        img_ind = src - traj_starts[traj_ind]
        end_img_ind = dest - traj_starts[traj_ind]
        print('trajectory', traj_ind, 'inds', img_ind, end_img_ind)
        gt_path = range(src, dest, 2)

        plot_traj(gt_path, f'Ground Truth Trajectory ({src}, {dest})', f'./figures/gt_{src}_{dest}.png')

        path, dist = dijkstras(src, dest, edges)
        plot_traj(path, f'Dijkstras Trajectory ({src}, {dest})', f'./figures/dijkstras_{src}_{dest}.png', show_score=True)
    except:
        pass



