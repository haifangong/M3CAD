import os
import torch
from Bio.PDB import PDBParser, is_aa
from Bio.PDB.Polypeptide import protein_letters_3to1
from torch.utils.data import random_split, Dataset
import torch.nn.functional as F
import random
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from torchvision import transforms
import torch.nn.utils.rnn as rnn_utils


def clamp(n, smallest, largest):
    return sorted([smallest, n, largest])[1]


ATOMS = {'H': 1, 'C': 12, 'N': 14, 'O': 16, 'S': 30}  # weight of the atom
ATOMS_R = {'H': 1, 'C': 1.5, 'N': 1.5, 'O': 1.5, 'S': 2}  # radius of the atom
AMINO_ACID_WATER = {'A': 255, 'V': 255, 'P': 255, 'F': 255, 'W': 255, 'I': 255, 'L': 255, 'G': 155, 'M': 155,
                    'Y': 55, 'S': 55, 'T': 55, 'C': 55, 'N': 55, 'Q': 55, 'D': 55, 'E': 55, 'K': 55, 'R': 55, 'H': 55}
AMINO_ACID_CHARGE = {'D': 55, 'E': 55, 'A': 155, 'V': 155, 'P': 155, 'F': 155, 'W': 155, 'I': 155, 'L': 155, 'G': 155,
                     'M': 155, 'Y': 155, 'S': 155, 'T': 155, 'C': 155, 'N': 155, 'Q': 155, 'K': 255, 'R': 255, 'H': 255}
AMAs = {'G': 20, 'A': 1, 'V': 2, 'L': 3, 'I': 4, 'P': 5, 'F': 6, 'Y': 7, 'W': 8, 'S': 9, 'T': 10, 'C': 11,
        'M': 12, 'N': 13, 'Q': 14, 'D': 15, 'E': 16, 'K': 17, 'R': 18, 'H': 19, 'X': 21}

# AMAs = {'G': 20, 'A': 1, 'V': 2, 'L': 3, 'I': 4, 'P': 5, 'F': 6, 'Y': 7, 'W': 8, 'S': 9, 'T': 10, 'C': 11,
#         'M': 12, 'N': 13, 'Q': 14, 'D': 15, 'E': 16, 'K': 17, 'R': 18, 'H': 19}


def train_valid_split(data_set, fold, mode):
    train_set, valid_set = [], []
    for i in range(len(data_set)):
        if (i + fold) % 5 != 0:
            train_set.append(data_set[i])
        else:
            valid_set.append(data_set[i])
    if mode == "train":
        return train_set
    elif mode == "valid":
        return valid_set


def collate_fn(batch):
    vox = torch.stack([(v[0]) for v in batch])
    xs = [(v[1]) for v in batch]
    # print(len(xs))
    # print(xs[0].shape)
    ys = torch.stack([v[2] for v in batch])
    # 获得每个样本的序列长度
    seq_lengths = torch.LongTensor([v for v in map(len, xs)])
    # 每个样本都padding到当前batch的最大长度
    xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    # 把xs和ys按照序列长度从大到小排序
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    # print(seq_lengths)
    xs = xs[perm_idx]
    ys = ys[perm_idx]
    vox = vox[perm_idx]
    return vox, xs, ys, seq_lengths

    #
    # data_tuple.sort(key=lambda x: len(x[1]), reverse=True)
    # seq = [sq[1] for sq in data_tuple]
    # vox = [sq[0] for sq in data_tuple]
    # label = [sq[2] for sq in data_tuple]
    # data_length = [len(sq) for sq in seq]
    # seq = rnn_utils.pad_sequence(seq, batch_first=True, padding_value=0.0)
    # return torch.tensor(np.asarray(vox)), torch.tensor(np.asarray(seq)), torch.tensor(np.asarray(label)), data_length


import torch
from Bio.PDB import PDBParser, DSSP
import mdtraj as md


def read_secondary_structure(pdb_id: str, chain_id: str = "A") -> str:
    pdb_file = f"{pdb_id}"
    traj = md.load(pdb_file)
    secondary_structure = md.compute_dssp(traj)[0]
    return "".join(secondary_structure)


def secondary_structure_to_fixed_tensor(secondary_structure: str, max_length: int = 30) -> torch.Tensor:
    sse_to_index = {"H": 1, "E": 2, "C": 3}
    num_sse = len(sse_to_index)
    fixed_tensor = torch.zeros(max_length)
    for i, sse in enumerate(secondary_structure):
        if i >= max_length:
            break
        fixed_tensor[i] = sse_to_index[sse]
    # fixed_tensor = torch.zeros(max_length, num_sse)
    # for i, sse in enumerate(secondary_structure):
    #     if i >= max_length:
    #         break
    #     index = sse_to_index[sse]
    #     fixed_tensor[i, index] = 1

    return fixed_tensor


# Test samples
# test_samples = ["1a3n", "1tim"]
#
# for pdb_id in test_samples:
#
#     print(f"Fixed-size secondary structure tensor for {pdb_id}:")
#     print(fixed_tensor)
#     print()
class ADataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''

    def __init__(self, mode='train', fold=0, task='anti'):
        self.data_list = []
        # all_data = pd.read_csv('metadata/data.csv', encoding="unicode_escape").values
        # all_data = pd.read_csv('metadata/mdr-regression.csv', encoding="unicode_escape").values
        # all_data = pd.read_csv('./metadata/toxin.csv', encoding="unicode_escape").values
        all_data = pd.read_csv('./metadata/data_pervious.csv', encoding="unicode_escape").values
        # 0-13 anti 2-13 specific [763 5619 5227 1156 4504 14 1561 0 9 95 0 138]
        # 14-20 toxic
        # 21-24 machine
        if task == 'anti' or task == 'mechanism' or task == 'mic' or task == 'anti-binary':
            # all_data = all_data[:14676, :]
            all_data = all_data[:12915, :]
            # all_data = all_data[:10389, :]
        elif task == 'toxin':
            all_data = all_data[23666:, :]
        idx_list, seq_list, labels = all_data[:, 0], all_data[:, 1], all_data[:, 2:]
        if task == 'anti':
            label_list = np.concatenate((labels[:, 2:7], labels[:, 8:9]), axis=1)
        elif task == 'toxin':
            # label_list = np.expand_dims(labels[:], axis=0)
            label_list_organ = labels[:, 15] + labels[:, 16] + labels[:, 17] + labels[:, 18] + labels[:, 19]
            # print(label_list_organ.shape)
            # print(labels[:, 20].shape)
            label_list_organ[label_list_organ > 0] = 1
            label_list = np.concatenate((label_list_organ[:], labels[:, 20]), axis=0)
            label_list = labels[:, 15:21]
        elif task == 'anti-all':
            label_list = labels[:, 2:14]
        elif task == 'mechanism':
            label_list = labels[:, 21:25]
        elif task == 'anti-binary':
            # this is dr classification
            label_list = labels[:, 2] + labels[:, 3] + labels[:, 4] + labels[:, 5] + labels[:, 6] + labels[:, 8]
            label_list[label_list > 0] = 1
        elif task == 'anti-regression':
            # this is mdr regression
            label_list = labels[:, 2] + labels[:, 5] + labels[:, 8] + labels[:, 11] + labels[:, 14] + labels[:, 21]
        elif task == 'mic':
            label_list = labels[:, 25]
            label_list[label_list > 512] = 512
            label_list[label_list > 1024] = -1
            label_list[label_list == 0] = -1
        else:
            label_list = labels

        # if type(label_list[0]) == float or type(label_list[0]) == str or type(label_list[0]) == int:
        #     self.num_classes = 1
        # else:
        self.num_classes = len(label_list[0])

        print('classes:', self.num_classes)
        print('class count:', np.sum(label_list, axis=0))
        print(len(idx_list))
        filter_data = []

        for idx in range(len(idx_list)):
            seq = seq_list[idx].upper().strip()
            # if label_list[idx] == -1:
            #     continue
            # if self.num_classes == 1:
            #     gt = [np.log2(label_list[idx]) if label_list[idx]>2 else 1]
            # else:
            # print(label_list[idx])  # [0 1 0 1 0 0]
            gt = [int(i) for i in label_list[idx]]  # Very Important [0, 1, 0, 1, 0, 0]
            # gt = [label_list[idx]]
            # print(gt)
            # return
            if 'X' in seq:
                continue
            elif 'O' in seq:
                continue
            elif 'U' in seq:
                continue
            # if len(seq) > 30:
            #     continue
            if len(seq) < 10:
                continue
            # gt[0] = 1 if gt[0] == 0 else 0
            # gt = [label_list[0][idx]]
            filter_data.append((seq, gt, str(idx_list[idx])))
        # filter_data.sort(key=lambda x: len(x[0]), reverse=True)
        # print(len(filter_data))
        spilt_data = filter_data if mode == 'all' else train_valid_split(filter_data, fold, mode)
        print(len(spilt_data))
        count = 0
        gt_list = []
        p = PDBParser()

        for item in tqdm(spilt_data):
            seq, gt, idx = item

            idx = idx.strip().split('_')[0]
            idx = idx.zfill(5)
            # print(seq)
            # if not os.path.exists("./pdb_gen/voxel/" + idx + ".pkl"):
            #     continue
            # voxel = pickle.load(open("./pdb_gen/voxel/" + idx + ".pkl", 'rb'))

            # assert os.path.exists("./pdb_gen/voxel/" + idx + ".pkl")
            # direct load
            # voxel = pickle.load(open("./pdb_gen/voxel/" + idx + ".pkl", 'rb'))
            if os.path.exists("./pdb_gen/pdb_dbassp/" + seq + "_relaxed_rank_1_model_3.pdb_gen"):
                voxel, seq_pdb = pdb_parser(p, idx, "./pdb_gen/pdb_dbassp/" + seq + "_relaxed_rank_1_model_3.pdb_gen")
                secondary_structure = read_secondary_structure("./pdb_gen/pdb_dbassp/" + seq + "_relaxed_rank_1_model_3.pdb_gen")
                secondary_structure_map = secondary_structure_to_fixed_tensor(secondary_structure)
            elif os.path.exists("./pdb_gen/pdb_dbassp/" + seq + "_real.pdb_gen"):
                voxel, seq_pdb = pdb_parser(p, idx, "./pdb_gen/pdb_dbassp/" + seq + "_real.pdb_gen")
                secondary_structure = read_secondary_structure("./pdb_gen/pdb_dbassp/" + seq + "_relaxed_rank_1_model_3.pdb_gen")
                secondary_structure_map = secondary_structure_to_fixed_tensor(secondary_structure)
            else:
                continue
                assert os.path.exists("./pdb_gen/pdb_dbassp/" + seq + "_real.pdb_gen")
            # assert seq == seq_pdb

            # normalize voxel data
            # voxel = np.zeros(voxel_ori.shape)
            # print(np.max(voxel_ori[0]))
            # print(np.max(voxel_ori[1]))
            # print(np.max(voxel_ori[2]))
            # voxel[0] = voxel_ori[0]/30
            # voxel[1] = voxel_ori[1]/3
            # voxel[2] = voxel_ori[2]/3
            # print(voxel.shape)
            seq_code = [AMAs[char] for char in seq]
            # seq_emb = F.one_hot(torch.tensor(seq_code).to(torch.int64), num_classes=21).to(torch.float16)
            seq_emb = [seq_code + [0] * (50 - len(seq_code))]
            # print(len(seq_emb))
            self.data_list.append((voxel, seq_emb, secondary_structure_map, gt))
            count += 1
            # seq_code2 = [AMAs[char] for char in seq[::-1]]
            # seq_emb2 = [seq_code2 + [0] * (50 - len(seq_code))]
            # self.data_list.append((voxel, seq_emb2, gt))

            # if gt[0] == 1:
            #     self.data_list.extend([(voxel, seq_emb, gt) for i in range(5])
            gt_list.append(gt)

        gt_list = np.asarray(gt_list)
        gt_class_wise_count = np.sum(gt_list, 0)
        print('class wise counts', gt_class_wise_count)
        print('missing counts:', count)
        print('used counts:', len(self.data_list))
        self.im_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90, expand=False, center=None),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            # transforms.RandomCrop(256),
            # transforms.Resize((512,512))
        ])

    def __getitem__(self, idx):
        voxel, seq, second_struct, gt = self.data_list[idx]
        # voxel[0] = voxel[0]
        # voxel[1] = voxel[1]/4 * 252
        # voxel[2] = voxel[2]/3 * 255
        voxel = self.im_aug(torch.Tensor(voxel).float())
        return torch.Tensor(voxel), torch.Tensor(seq), torch.Tensor(second_struct), torch.Tensor(gt), len(seq[0])

    def __len__(self):
        return len(self.data_list)


# a = ADataset()


class PDataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''

    def __init__(self, mode='train', fold=0, task='anti'):
        self.data_list = []
        all_data = pd.read_csv('./metadata/pretrain.csv', encoding="unicode_escape").values
        # all_data = pd.read_csv('./metadata/toxin.csv', encoding="unicode_escape").values
        # 0-13 anti 2-13 specific [763 5619 5227 1156 4504 14 1561 0 9 95 0 138]
        # 14-20 toxic
        # 21-24 machine
        idx_list, seq_list, labels = all_data[:, 0], all_data[:, 1], all_data[:, 2:]
        if task == 'anti':
            label_list = np.concatenate((labels[:, 2:7], labels[:, 8:9]), axis=1)
        elif task == 'toxin':
            # label_list = np.expand_dims(labels[:], axis=0)
            label_list = labels
        elif task == 'anti-all':
            label_list = labels[:, 2:14]
        elif task == 'mechanism':
            label_list = labels[:, 21:]
        else:
            label_list = labels
        print('classes:', len(label_list[0]))
        print('class count:', np.sum(label_list, axis=0))
        print(len(idx_list))
        filter_data = []

        for idx in range(len(idx_list)):
            seq = seq_list[idx].upper().strip()
            gt = [int(i) for i in label_list[idx]]
            # gt = [label_list[0][idx]]
            filter_data.append((seq, gt, str(idx_list[idx])))
        # print(len(filter_data))
        spilt_data = filter_data if mode == 'all' else train_valid_split(filter_data, fold, mode)
        print(len(spilt_data))
        count = 0
        for item in tqdm(spilt_data):
            seq, gt, idx = item
            idx = idx.strip().split('_')[0]
            idx = idx.zfill(5)
            assert os.path.exists("./pdb_gen/voxel/" + idx + ".pkl")
            voxel = pickle.load(open("./pdb_gen/voxel/" + idx + ".pkl", 'rb'))
            seq_code = [AMAs[char] for char in seq]
            seq_emb = [seq_code + [0] * (50 - len(seq_code))]
            self.data_list.append((voxel, seq_emb, gt))

            # if gt[0] == 1:
            #     self.data_list.extend([(voxel, seq_emb, gt) for i in range(5])

        print('missing counts:', count)
        print('used counts:', len(self.data_list))
        self.im_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90, expand=False, center=None),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            # transforms.RandomCrop(256),
            # transforms.Resize((512,512))
        ])

    def __getitem__(self, idx):
        voxel, seq, gt = self.data_list[idx]
        voxel = self.im_aug(torch.Tensor(voxel).float())
        return voxel, torch.Tensor(seq), torch.Tensor(gt)

    def __len__(self):
        return len(self.data_list)


def pdb_parser(parser, idx, pdb_path):
    """

    """
    voxel = np.zeros((3, 64, 64, 64), dtype=np.int8)
    structure = parser.get_structure(idx, pdb_path)
    id = ''
    seq_str = ''
    for i in structure[0]:
        id = i.id
    chain = structure[0][id]
    for res in chain:
        if is_aa(res.get_resname(), standard=True):
            resname = res.get_resname()
            amino = protein_letters_3to1[resname]
            seq_str += str(amino)
            ATOM_WATER = AMINO_ACID_WATER[amino]
            ATOM_CHARGE = AMINO_ACID_CHARGE[amino]
            ATOM_CATEGORY = AMAs[amino] * 20

            for i in res:
                if i.id not in ATOMS.keys():
                    continue
                x, y, z = i.get_coord()
                if abs(x) > 32:
                    x = clamp(x, -31, 31)
                if abs(y) > 32:
                    y = clamp(x, -31, 31)
                if abs(z) > 32:
                    z = clamp(x, -31, 31)
                x_i, y_i, z_i = int(x) + 32, int(y) + 32, int(z) + 32
                ATOM_WEIGHT = ATOMS[i.id]
                ATOM_R = ATOMS_R[i.id]
                # if ATOM_R == 1:
                #     voxel[0, x_i - 1:x_i, y_i - 1:y_i, z_i - 1:z_i] = ATOM_WEIGHT
                #     voxel[1, x_i - 1:x_i, y_i - 1:y_i, z_i - 1:z_i] = ATOM_WATER
                #     voxel[2, x_i - 1:x_i, y_i - 1:y_i, z_i - 1:z_i] = ATOM_CHARGE
                #     voxel[3, x_i - 1:x_i, y_i - 1:y_i, z_i - 1:z_i] = ATOM_CATEGORY
                # el
                if ATOM_R <= 1.5:
                    voxel[0, x_i - 1:x_i + 1, y_i - 1:y_i + 1, z_i - 1:z_i + 1] = ATOM_WEIGHT
                    voxel[1, x_i - 1:x_i + 1, y_i - 1:y_i + 1, z_i - 1:z_i + 1] = ATOM_WATER
                    voxel[2, x_i - 1:x_i + 1, y_i - 1:y_i + 1, z_i - 1:z_i + 1] = ATOM_CHARGE
                    # voxel[3, x_i - 1:x_i + 1, y_i - 1:y_i + 1, z_i - 1:z_i + 1] = ATOM_CATEGORY
                else:
                    voxel[0, x_i - ATOM_R: x_i + ATOM_R, x_i - ATOM_R: x_i + ATOM_R,
                    x_i - ATOM_R: x_i + ATOM_R] = ATOM_WEIGHT
                    voxel[1, x_i - ATOM_R: x_i + ATOM_R, x_i - ATOM_R: x_i + ATOM_R,
                    x_i - ATOM_R: x_i + ATOM_R] = ATOM_WATER
                    voxel[2, x_i - ATOM_R: x_i + ATOM_R, x_i - ATOM_R: x_i + ATOM_R,
                    x_i - ATOM_R: x_i + ATOM_R] = ATOM_CHARGE
                    # voxel[3, x_i - ATOM_R: x_i + ATOM_R, x_i - ATOM_R: x_i + ATOM_R,
                    # x_i - ATOM_R: x_i + ATOM_R] = ATOM_CATEGORY
    return voxel, seq_str


class MDataset(Dataset):
    """
    mutation dataset
    """

    def __init__(self, mode='train', fold=0, task='anti'):
        self.data_list = []
        all_data = pd.read_csv('metadata/mutated-ood.csv', encoding="unicode_escape").values
        idx_list, seq_list = all_data[:, 0], all_data[:, 1]
        print(len(seq_list))
        p = PDBParser()
        for idx in range(6, 7):
            seq_wide = seq_list[idx * 2 - 2]
            seq_mutate = seq_list[idx * 2 - 1]
            idx = str(idx).zfill(2)
            print(seq_wide)
            print(seq_mutate)
            # pdb_path_wide = "./pdb_gen/mutated_pdbs/W" + idx + "_relaxed_rank_1_model_3.pdb_gen"
            # pdb_path_mutated = "./pdb_gen/mutated_pdbs/M" + idx + "_relaxed_rank_1_model_3.pdb_gen"
            pdb_path_wide = './pdb_gen/relaxed_pdbs/NO11415_relaxed_rank_1_model_3.pdb_gen'
            pdb_path_mutated = './pdb_gen/relaxed_pdbs/NO11415_relaxed_rank_1_model_3.pdb_gen'
            assert os.path.exists(pdb_path_wide)
            assert os.path.exists(pdb_path_mutated)

            seq_emb_wide = [[AMAs[char] for char in seq_wide] + [0] * (50 - len(seq_wide))]
            seq_emb_mutate = [[AMAs[char] for char in seq_mutate] + [0] * (50 - len(seq_mutate))]

            voxel_mutated, seq_pdb = pdb_parser(p, idx, pdb_path_mutated)
            voxel_wide, seq_pdb = pdb_parser(p, idx, pdb_path_wide)

            mutate = (voxel_mutated, seq_emb_mutate)
            wide = (voxel_wide, seq_emb_wide)

            self.data_list.append((wide, mutate))
        print('used counts:', len(self.data_list))

    def __getitem__(self, idx):
        mutate, wide = self.data_list[idx]
        return (torch.Tensor(mutate[0]).float(), torch.Tensor(mutate[1])), (
            torch.Tensor(wide[0]).float(), torch.Tensor(wide[1]))

    def __len__(self):
        return len(self.data_list)


class GSDataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''

    def __init__(self, path='./gendata/gen20k.csv'):
        self.data_list = []
        f = open(path, 'r')
        lines = f.readlines()[1:]
        all_data = pd.read_csv(path, encoding="unicode_escape").values
        idx_list, seq_list = all_data[:, 0], all_data[:, 1]
        filter_data = []
        for idx in range(1, len(seq_list)):
            # print((idx, seq_list[idx-1]))
            filter_data.append((idx, seq_list[idx - 1]))
        count = 0
        for item in tqdm(filter_data):
            index, seq = item
            seq_code = [AMAs[char] for char in seq]
            seq_emb = [seq_code + [0] * (50 - len(seq_code))]
            count += 1
            self.data_list.append((seq_emb, lines[count], index))

        print('used counts:', len(self.data_list))

    def __getitem__(self, idx):
        seq_emb, context, index = self.data_list[idx]
        return torch.Tensor(seq_emb), context, index

    def __len__(self):
        return len(self.data_list)


class GDataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''

    def __init__(self, path='gendata/v2_filter_r3.csv', mode='dl'):
        self.data_list = []
        self.mode = mode
        f = open(path, 'r')
        lines = f.readlines()[1:]
        all_data = pd.read_csv(path, encoding="unicode_escape").values
        idx_list, seq_list = all_data[:, 0], all_data[:, 1]
        filter_data = []
        for idx in range(1, len(seq_list) + 1):
            filter_data.append((idx, seq_list[idx - 1]))
        p = PDBParser()

        count = 0
        for item in tqdm(filter_data):
            index, seq = item
            idx = str(index).zfill(3)
            if not os.path.exists("./gendata/gen_v3_20k/" + seq + "_relaxed_rank_1_model_3.pdb_gen"):
                print(seq)
                count += 1
                continue
            assert os.path.exists("./gendata/gen_v3_20k/" + seq + "_relaxed_rank_1_model_3.pdb_gen")

            seq_code = [AMAs[char] for char in seq]
            seq_emb = [seq_code + [0] * (50 - len(seq_code))]

            voxel, seq_pdb = pdb_parser(p, idx, "./gendata/gen_v3_20k/" + seq + "_relaxed_rank_1_model_3.pdb_gen")
            # print(lines[count])
            if mode == "prior":
                self.data_list.append(
                    ("./gendata/gen_v3_20k/" + seq + "_relaxed_rank_1_model_3.pdb_gen", seq, lines[count], index))
            else:
                self.data_list.append((voxel, seq_emb, lines[count], index))
            count += 1

        print('used counts:', len(self.data_list))

    def __getitem__(self, idx):
        if self.mode == "dl":
            voxel, seq, gt, index = self.data_list[idx]
            return torch.Tensor(voxel).float(), torch.Tensor(seq), gt, index
        else:
            voxel, seq, gt, index = self.data_list[idx]
            return voxel, seq, gt, index

    def __len__(self):
        return len(self.data_list)


class HDataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''

    def __init__(self, mode='train', fold=0, task='mechanism'):
        self.data_list = []
        all_data = pd.read_csv('./metadata/human.csv', encoding="unicode_escape").values
        # all_data = pd.read_csv('./metadata/toxin.csv', encoding="unicode_escape").values
        # 0-13 anti 2-13 specific [763 5619 5227 1156 4504 14 1561 0 9 95 0 138]
        # 14-20 toxic
        # 21-24 machine
        idx_list, seq_list = all_data[:, 0], all_data[:, 1]

        print(len(idx_list))
        filter_data = []

        for idx in range(len(idx_list)):
            seq = seq_list[idx].upper().strip()
            # if idx < 134:
            #     print(len(seq))
            gt = [0] if idx < 134 else [1]
            if idx > 300:
                break
            filter_data.append((seq, gt, str(idx_list[idx])))
        # print(len(filter_data))
        spilt_data = filter_data if mode == 'all' else train_valid_split(filter_data, fold, mode)
        print(len(spilt_data))
        count = 0
        for item in tqdm(spilt_data):
            seq, gt, idx = item
            voxel = torch.zeros((64, 64, 64))
            seq_code = [AMAs[char] for char in seq][:500]
            seq_emb = [seq_code + [0] * (500 - len(seq_code))]
            self.data_list.append((voxel, seq_emb, gt))

        print('missing counts:', count)
        print('used counts:', len(self.data_list))
        self.im_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90, expand=False, center=None),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            # transforms.RandomCrop(256),
            # transforms.Resize((512,512))
        ])

    def __getitem__(self, idx):
        voxel, seq, gt = self.data_list[idx]
        voxel = self.im_aug(torch.Tensor(voxel).float())
        return voxel, torch.Tensor(seq), torch.Tensor(gt)

    def __len__(self):
        return len(self.data_list)

# h = MDataset()
