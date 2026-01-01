from collections import Counter
import os
import torch
import networkx as nx
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

import numpy as np
from scipy.ndimage import gaussian_filter1d, convolve1d
from scipy.signal.windows import triang
import pickle


def cosine_similarity(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
            map(laplace, np.arange(-half_ks, half_ks + 1)))
    return kernel_window


def get_bin_idx(x):
    return max(min(int(x * np.float32(5)), 12), -12)


class PairData(Data):
    def __init__(self, edge_index_s, x_s):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s

    def __inc__(self, key, value, *args):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'wide_nodes':
            return self.x_s.num_nodes
        else:
            return super().__inc__(key, value, *args)


import pandas as pd


def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size],
                                        generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


def select_feat(train_data, valid_data):
    y_train, y_valid = train_data[:, 1:], valid_data[:, 1:]
    x_train, x_valid = train_data[:, 0], valid_data[:, 0]

    return x_train, x_valid, y_train, y_valid


def obtain_dict(filepath='/data2/gonghaifan/codes/graph/Data/peptides-complete.csv'):
    file = pd.read_csv(filepath, encoding="unicode_escape").values
    dict_aa = {}
    for line in file:
        idx = line[0]
        if type(line[4]) == float:
            continue
        seq = line[4].upper().strip()
        dict_aa[seq] = idx
    return dict_aa


import random


def load_graph_dataset(graph_dir='', split="train", labeled=True, dir=False):
    data_list = []
    num_nodes = 0
    num_edges = 0
    seq_dict = obtain_dict()
    train_data = pd.read_csv('./Data/data.csv').values
    train_data, valid_data = train_valid_split(train_data, 0.01, 1234)
    x_train, x_valid, y_train, y_valid = select_feat(train_data, valid_data)
    # print(x_train)
    # if split == "train":
    print(len(x_train))
    name_list, gt_list = [], []
    for idx in range(len(x_train)):
        x = x_train[idx]
        if len(x.upper()) > 50:
            continue
        if len(x.upper()) < 10:
            continue
        name_list.append(x.upper().strip())
        y = y_train[idx]
        gt_list.append([int(i) for i in y])

    # count_list = [0 for i in range(5)]
    # for i in gt_list:
    # print(gt_list)
    print(np.sum(np.array(gt_list), 0))
    id_list = []
    new_name_list = []
    for name in name_list:
        if name not in seq_dict.keys():
            continue
        else:
            new_name_list.append(name)
            id_list.append(str(seq_dict[name]).zfill(5))
    print('seq:', len(id_list))
    print('gt:', len(gt_list))
    # print(gt_list)
    count = 0
    AMAs = {'G': 0, 'A': 1, 'V': 2, 'L': 3, 'I': 4, 'P': 5, 'F': 6, 'Y': 7, 'W': 8, 'S': 9, 'T': 10, 'C': 11,
            'M': 12, 'N': 13, 'Q': 14, 'D': 15, 'E': 16, 'K': 17, 'R': 18, 'H': 19, 'X': 20}
    count_1 = 0
    for name in id_list:
        # print(sum(gt_list[count]))
        if sum(gt_list[count]) == 0:
            count += 1
            continue
        else:
            # pass
            # gt_list[count][1]=0
            # gt_list[count][2]=0
            # if count % 2 == 0:
            #     gt_list[count][0] = 1
            # else:
            #     gt_list[count][0] = 0

            if gt_list[count][0] == 1 and gt_list[count][1] == 0 and gt_list[count][2] == 0:
                # print(gt_list[count])
                count_1 += 1
                if random.random() > 0.5:
                    count += 1
                    continue

        name = name.strip().split(',')[0].zfill(5)
        if not os.path.exists(f"/data2/gonghaifan/codes/graph/Data/graph/{name}.pkl"):
            continue
        G_wt = nx.read_gpickle(f"/data2/gonghaifan/codes/graph/Data/graph/{name}.pkl")
        data_wt = from_networkx(G_wt)

        wt_node_count = data_wt.num_nodes

        data_direct = PairData(data_wt.edge_index, data_wt.x)
        seq = new_name_list[count]
        seq_emb = [AMAs[char] for char in seq]
        data_direct.seq = torch.Tensor([seq_emb + [0] * (50 - len(seq_emb))])
        data_direct.wt_count = wt_node_count

        if labeled:
            data_direct.y = gt_list[count]
        count += 1
        data_list.append(data_direct)
        num_nodes += data_wt.num_nodes
        num_edges += data_wt.num_edges
    print('dataset', count)
    print(f'{split.upper()} DATASET:')
    print(f'Number of nodes: {num_nodes / len(data_list):.2f}')
    print(f'Number of edges: {num_edges / len(data_list):.2f}')
    print(f'Average node degree: {num_edges / num_nodes:.2f}')
    print(count_1)
    return data_list


# load_dataset()
def load_dataset(graph_dir='', split="train", labeled=True, dir=False):
    data_list = []
    num_nodes = 0
    num_edges = 0
    seq_dict = obtain_dict()
    train_data = pd.read_csv('./Data/data.csv').values
    train_data, valid_data = train_valid_split(train_data, 0.01, 1234)
    x_train, x_valid, y_train, y_valid = select_feat(train_data, valid_data)
    # print(x_train)
    # if split == "train":
    print(len(x_train))
    name_list, gt_list = [], []
    for idx in range(len(x_train)):
        x = x_train[idx]
        if len(x.upper()) > 50:
            continue
        if len(x.upper()) < 10:
            continue
        name_list.append(x.upper().strip())
        y = y_train[idx]
        gt_list.append([int(i) for i in y])

    # count_list = [0 for i in range(5)]
    # for i in gt_list:
    # print(gt_list)
    print(np.sum(np.array(gt_list), 0))
    id_list = []
    new_name_list = []
    for name in name_list:
        if name not in seq_dict.keys():
            continue
        else:
            new_name_list.append(name)
            id_list.append(str(seq_dict[name]).zfill(5))
    print('seq:', len(id_list))
    print('gt:', len(gt_list))
    # print(gt_list)
    count = 0
    AMAs = {'G': 0, 'A': 1, 'V': 2, 'L': 3, 'I': 4, 'P': 5, 'F': 6, 'Y': 7, 'W': 8, 'S': 9, 'T': 10, 'C': 11,
            'M': 12, 'N': 13, 'Q': 14, 'D': 15, 'E': 16, 'K': 17, 'R': 18, 'H': 19, 'X': 20}
    count_1 = 0
    for name in id_list:
        # print(sum(gt_list[count]))
        if sum(gt_list[count]) == 0:
            count += 1
            continue
        else:
            # pass
            # gt_list[count][1]=0
            # gt_list[count][2]=0
            # if count % 2 == 0:
            #     gt_list[count][0] = 1
            # else:
            #     gt_list[count][0] = 0

            if gt_list[count][0] == 1 and gt_list[count][1] == 0 and gt_list[count][2] == 0:
                # print(gt_list[count])
                count_1 += 1
                if random.random() > 0.5:
                    count += 1
                    continue

        name = name.strip().split(',')[0].zfill(5)
        if not os.path.exists(f"/data2/gonghaifan/codes/graph/Data/voxel/{name}.pkl"):
            continue
        voxel = pickle.load(f"/data2/gonghaifan/codes/graph/Data/voxel/{name}.pkl")

        seq = new_name_list[count]
        seq_emb = [AMAs[char] for char in seq]
        data_direct.seq = torch.Tensor([seq_emb + [0] * (50 - len(seq_emb))])
        data_direct.wt_count = wt_node_count

        gt = gt_list[count]

        data_list.append(data_direct)

    return data_list
