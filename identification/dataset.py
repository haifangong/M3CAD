import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from scipy.ndimage import map_coordinates
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from Bio.PDB import PDBParser, is_aa
from Bio.PDB.Polypeptide import protein_letters_3to1
import mdtraj as md
from Bio import pairwise2

# --- Constants ---
ATOMS = {'H': 1, 'C': 12, 'N': 14, 'O': 16, 'S': 30}
ATOMS_R = {'H': 1, 'C': 1.5, 'N': 1.5, 'O': 1.5, 'S': 2}
AMINO_ACID_WATER = {
    'A': 255, 'V': 255, 'P': 255, 'F': 255, 'W': 255, 'I': 255, 'L': 255, 'G': 155, 'M': 155,
    'Y': 55, 'S': 55, 'T': 55, 'C': 55, 'N': 55, 'Q': 55, 'D': 55, 'E': 55, 'K': 55, 'R': 55, 'H': 55
}
AMINO_ACID_CHARGE = {
    'D': 55, 'E': 55, 'A': 155, 'V': 155, 'P': 155, 'F': 155, 'W': 155, 'I': 155, 'L': 155, 'G': 155,
    'M': 155, 'Y': 155, 'S': 155, 'T': 155, 'C': 155, 'N': 155, 'Q': 155, 'K': 255, 'R': 255, 'H': 255
}
AMAs = {
    'G': 20, 'A': 1, 'V': 2, 'L': 3, 'I': 4, 'P': 5, 'F': 6, 'Y': 7, 'W': 8, 'S': 9, 'T': 10, 'C': 11,
    'M': 12, 'N': 13, 'Q': 14, 'D': 15, 'E': 16, 'K': 17, 'R': 18, 'H': 19, 'X': 21
}

# --- Utility Functions ---

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def get_split_indices(total_count, fold, mode, json_path="dataset_split.json"):
    utils_dir = "utils"
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)
    
    full_path = os.path.join(utils_dir, json_path)
    
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            indices_dict = json.load(f)
    else:
        all_indices = list(range(total_count))
        train_indices = [i for i in all_indices if (i + fold) % 5 != 0]
        valid_indices = [i for i in all_indices if (i + fold) % 5 == 0]
        indices_dict = {"train": train_indices, "valid": valid_indices}
        with open(full_path, 'w') as f:
            json.dump(indices_dict, f)
    return indices_dict[mode] if mode in indices_dict else indices_dict.get("train")

def pdb_parser(parser, idx, pdb_path):
    # Fix: Use uint8 to prevent OverflowError (0-255)
    voxel = np.zeros((3, 64, 64, 64), dtype=np.uint8)
    try:
        structure = parser.get_structure(idx, pdb_path)
        chain = next(structure[0].get_chains())
        seq_str = ''
        
        for res in chain:
            if is_aa(res.get_resname(), standard=True):
                resname = res.get_resname()
                amino = protein_letters_3to1.get(resname, 'X')
                seq_str += str(amino)
                
                atom_water = AMINO_ACID_WATER.get(amino, 0)
                atom_charge = AMINO_ACID_CHARGE.get(amino, 0)

                for atom in res:
                    if atom.id not in ATOMS:
                        continue
                    coord = atom.get_coord()
                    x_i = int(clamp(coord[0], -31, 31) + 32)
                    y_i = int(clamp(coord[1], -31, 31) + 32)
                    z_i = int(clamp(coord[2], -31, 31) + 32)
                    
                    radius = ATOMS_R[atom.id]
                    r_range = 1 if radius <= 1.5 else int(radius)
                    
                    z_slice = slice(max(0, z_i-r_range), min(64, z_i+r_range))
                    y_slice = slice(max(0, y_i-r_range), min(64, y_i+r_range))
                    x_slice = slice(max(0, x_i-r_range), min(64, x_i+r_range))
                    
                    voxel[0, x_slice, y_slice, z_slice] = ATOMS[atom.id]
                    voxel[1, x_slice, y_slice, z_slice] = atom_water
                    voxel[2, x_slice, y_slice, z_slice] = atom_charge
        return voxel, seq_str
    except Exception:
        return voxel, ""

# --- Dataset Classes ---

class ADataset(Dataset):
    def __init__(self, mode='train', fold=0, task='anti'):
        self.data_list = []
        # Load CSV and force numeric conversion where possible to avoid object types
        df = pd.read_csv('metadata/data_processed.csv', encoding="unicode_escape")
        all_data = df.values
        
        if task in ['anti', 'mechanism', 'mic', 'anti-binary']:
            all_data = all_data[:12915, :]
        elif task == 'toxin':
            all_data = all_data[23666:, :]
        elif task == 'regression':
            # Use all data
            pass
            
        idx_list, seq_list, labels = all_data[:, 0], all_data[:, 1], all_data[:, 2:]
        
        if task == 'anti':
            label_list = np.concatenate((labels[:, 2:7], labels[:, 8:9]), axis=1)
        elif task == 'toxin':
            label_list = labels[:, 15:21]
        elif task == 'anti-all':
            label_list = labels[:, 2:14]
        elif task == 'mechanism':
            label_list = labels[:, 21:25]
        elif task == 'mic':
            label_list = labels[:, 25]
            label_list = np.where(label_list > 512, 512, label_list)
            label_list = np.where((label_list > 1024) | (label_list == 0), -1, label_list)
        elif task == 'regression':
            label_list = labels[:, 1]  # all species num
        else:
            label_list = labels

        # Fix: Ensure label_list is numeric (float32) to avoid object-type errors in torch.tensor
        label_list = label_list.astype(np.float32)
        
        # Calculate number of classes based on label_list shape
        if label_list.ndim == 1:
            self.num_classes = 1
        else:
            self.num_classes = label_list.shape[1]
        
        filter_data = []
        for i in range(len(idx_list)):
            seq = str(seq_list[i]).upper().strip()
            if any(x in seq for x in ['X', 'O', 'U']) or len(seq) < 10:
                continue
            # Store label as a clean numpy array
            filter_data.append((seq, label_list[i], str(idx_list[i])))

        split_indices = get_split_indices(len(filter_data), fold, mode, f"split_{task}_{fold}.json")
        split_data = [filter_data[i] for i in split_indices] if mode != 'all' else filter_data

        p = PDBParser(QUIET=True)
        for seq, gt, idx_str in tqdm(split_data, desc=f"Loading {mode}"):
            idx_clean = idx_str.strip().split('_')[0].zfill(5)
            paths = [f"./pdb/pdbs/{seq}_relaxed_rank_1_model_3.pdb", f"./pdb/pdb_dbassp/{seq}_real.pdb"]
            found_path = next((path for path in paths if os.path.exists(path)), None)
            
            if found_path:
                voxel, _ = pdb_parser(p, idx_clean, found_path)
                seq_code = [AMAs.get(char, 21) for char in seq]
                # Pad sequence to 50
                seq_padded = seq_code[:50] + [0] * max(0, 50 - len(seq_code))
                # Store as float32/int64 numpy arrays to ensure torch compatibility
                self.data_list.append((voxel, np.array([seq_padded], dtype=np.int64), 0, gt, len(seq)))

        self.im_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90)
        ])

    def __getitem__(self, idx):
        voxel, seq, second_struct, gt, seq_len = self.data_list[idx]
        # Convert voxel to float for augmentation and model input
        voxel_tensor = self.im_aug(torch.from_numpy(voxel).float())
        # Fix: Convert items to tensors explicitly ensuring they aren't object-type
        return (
            voxel_tensor, 
            torch.from_numpy(seq), 
            torch.tensor(second_struct, dtype=torch.long), 
            torch.tensor(gt), 
            seq_len
        )

    def __len__(self):
        return len(self.data_list)

class PDataset(Dataset):
    def __init__(self, mode='train', fold=0, task='anti'):
        self.data_list = []
        all_data = pd.read_csv('./metadata/pretrain.csv', encoding="unicode_escape").values
        idx_list, seq_list, labels = all_data[:, 0], all_data[:, 1], all_data[:, 2:]
        
        if task == 'anti': 
            label_list = np.concatenate((labels[:, 2:7], labels[:, 8:9]), axis=1)
        else:
            label_list = labels
        
        label_list = label_list.astype(np.float32)

        filter_data = []
        for i in range(len(idx_list)):
            seq = str(seq_list[i]).upper().strip()
            filter_data.append((seq, label_list[i], str(idx_list[i])))
            
        split_indices = get_split_indices(len(filter_data), fold, mode, "split_pretrain.json")
        split_data = [filter_data[i] for i in split_indices]

        for seq, gt, idx_str in tqdm(split_data):
            idx_clean = idx_str.strip().split('_')[0].zfill(5)
            pkl_path = f"./pdb/voxel/{idx_clean}.pkl"
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    voxel = pickle.load(f)
                seq_code = [AMAs.get(char, 21) for char in seq]
                seq_padded = seq_code[:50] + [0] * max(0, 50 - len(seq_code))
                self.data_list.append((voxel, np.array(seq_padded, dtype=np.int64), gt))

        self.im_aug = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(90)])

    def __getitem__(self, idx):
        voxel, seq, gt = self.data_list[idx]
        return self.im_aug(torch.from_numpy(voxel).float()), torch.from_numpy(seq), torch.from_numpy(gt)

    def __len__(self):
        return len(self.data_list)

class GSDataset(Dataset):
    def __init__(self, path='./gendata/gen20k.csv'):
        self.data_list = []
        df = pd.read_csv(path, encoding="unicode_escape")
        with open(path, 'r') as f:
            lines = f.readlines()[1:]
            
        for i, row in tqdm(df.iterrows(), total=len(df)):
            seq = str(row[1])
            seq_code = [AMAs.get(c, 21) for c in seq]
            seq_emb = [seq_code + [0] * (50 - len(seq_code))]
            self.data_list.append((seq_emb, lines[i], row[0]))

    def __getitem__(self, idx):
        seq_emb, context, index = self.data_list[idx]
        return torch.tensor(seq_emb), context, index

    def __len__(self):
        return len(self.data_list)

class GDataset(Dataset):
    def __init__(self, path='gendata/v3_top20k.csv', mode='dl'):
        self.data_list = []
        df = pd.read_csv(path, encoding="unicode_escape")
        with open(path, 'r') as f:
            lines = f.readlines()[1:]
            
        for i, row in tqdm(df.iterrows(), total=len(df)):
            seq = str(row[1])
            seq_code = [AMAs.get(c, 21) for c in seq]
            seq_emb = np.array([seq_code + [0] * (50 - len(seq_code))])
            self.data_list.append((0, seq_emb, lines[i], seq))

    def __getitem__(self, idx):
        voxel, seq, gt, index = self.data_list[idx]
        return torch.tensor(voxel).float(), torch.tensor(seq), gt, index

    def __len__(self):
        return len(self.data_list)

class HDataset(Dataset):
    def __init__(self, mode='train', fold=0, task='mechanism'):
        self.data_list = []
        all_data = pd.read_csv('./metadata/human.csv', encoding="unicode_escape").values
        idx_list, seq_list = all_data[:, 0], all_data[:, 1]

        filter_data = []
        for i in range(min(len(idx_list), 300)):
            seq = str(seq_list[i]).upper().strip()
            gt = [0] if i < 134 else [1]
            filter_data.append((seq, gt, str(idx_list[i])))
            
        split_indices = get_split_indices(len(filter_data), fold, mode, "split_human.json")
        split_data = [filter_data[i] for i in split_indices]

        for seq, gt, _ in tqdm(split_data):
            voxel = np.zeros((64, 64, 64))
            seq_code = [AMAs.get(c, 21) for c in seq][:500]
            seq_emb = [seq_code + [0] * (500 - len(seq_code))]
            self.data_list.append((voxel, seq_emb, gt))

        self.im_aug = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(90)])

    def __getitem__(self, idx):
        voxel, seq, gt = self.data_list[idx]
        return self.im_aug(torch.from_numpy(voxel).float().unsqueeze(0)), torch.tensor(seq), torch.tensor(gt)

    def __len__(self):
        return len(self.data_list)

# --- Visualization ---

def visualize_voxel_3d_rgb(pdb_path, output_jpg, iso_channel=0, threshold=0.5):
    parser = PDBParser(QUIET=True)
    voxel, _ = pdb_parser(parser, 'struct', pdb_path)
    volumes = [voxel[i].astype(np.float32) for i in range(3)]
    verts, faces, normals, values = measure.marching_cubes(volumes[iso_channel], level=threshold)
    coords = np.vstack([verts[:, 2], verts[:, 1], verts[:, 0]])
    sampled = np.zeros((verts.shape[0], 3), dtype=np.float32)
    for ch in range(3):
        sampled[:, ch] = map_coordinates(volumes[ch], coords, order=1)
    face_vals = sampled[faces].mean(axis=1)
    face_colors = np.zeros((faces.shape[0], 4))
    for ch in range(3):
        v_min, v_max = volumes[ch].min(), volumes[ch].max()
        face_colors[:, ch] = (face_vals[:, ch] - v_min) / (v_max - v_min + 1e-5)
    face_colors[:, 3] = 1.0
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], facecolors=face_colors, linewidth=0.05, edgecolor='gray')
    ax.add_collection3d(mesh)
    ax.set_xlim(0, 64); ax.set_ylim(0, 64); ax.set_zlim(0, 64)
    plt.savefig(output_jpg, dpi=300)
    plt.close()

def visualize_data_distribution():
    train_counts = [689, 4814, 4487, 848, 3887, 1427]
    valid_counts = [101, 736, 675, 114, 643, 236]
    classes = np.arange(len(train_counts))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(classes - width/2, train_counts, width, label='Train')
    ax.bar(classes + width/2, valid_counts, width, label='Valid')
    ax.set_xticks(classes)
    ax.set_xticklabels([f"C{i}" for i in classes])
    ax.legend()
    plt.savefig('class_distribution.jpg', dpi=300)
    plt.show()

def stat_property(gen_path, train_path='metadata/data_processed.csv'):
    gen_seqs = pd.read_csv(gen_path, encoding="unicode_escape").iloc[:, 1].astype(str).tolist()
    train_seqs = pd.read_csv(train_path, encoding="unicode_escape").head(12915).iloc[:, 1].astype(str).tolist()

    def get_ss(seqs):
        ah, bs = [], []
        for s in tqdm(seqs[:500], desc="DSSP"):
            p = next((f for f in [f"./pdb/pdbs/{s}_relaxed_rank_1_model_3.pdb", f"./pdb/pdb_dbassp/{s}_real.pdb"] if os.path.exists(f)), None)
            if p:
                dssp = md.compute_dssp(md.load(p))[0]
                ah.append(np.mean(dssp == 'H')); bs.append(np.mean(dssp == 'E'))
        return ah, bs

    g_ah, g_bs = get_ss(gen_seqs); t_ah, t_bs = get_ss(train_seqs)
    nw, sw = [], []
    for g in tqdm(gen_seqs[:100], desc="Align"):
        scores = [(pairwise2.align.globalxx(g, t, score_only=True)/max(len(g), len(t)), 
                   pairwise2.align.localxx(g, t, score_only=True)/max(len(g), len(t))) for t in train_seqs]
        nw.append(max(s[0] for s in scores)); sw.append(max(s[1] for s in scores))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot([g_ah, t_ah], labels=['Gen', 'Train']); axes[0].set_title('Alpha-Helix')
    axes[1].boxplot([g_bs, t_bs], labels=['Gen', 'Train']); axes[1].set_title('Beta-Strand')
    plt.savefig('ss_comp.jpg'); plt.close()

    # Convert similarity scores to percentages and create histogram with 1% bins
    nw_percent = [s * 100 for s in nw]  # Convert to percentage
    sw_percent = [s * 100 for s in sw]  # Convert to percentage
    
    # Create bins with 1% intervals (0 to 100%)
    bins = np.arange(0, 101, 1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(nw_percent, bins=bins, alpha=0.7, label='Needleman-Wunsch', color='lightblue', edgecolor='black', linewidth=0.5)
    plt.hist(sw_percent, bins=bins, alpha=0.7, label='Smith-Waterman', color='#FFCC99', edgecolor='black', linewidth=0.5)  # Light orange
    plt.xlabel('Similarity (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Sequence Similarity Distribution', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.xlim(0, 100)
    plt.savefig('sim_dist.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gen AH: {np.mean(g_ah):.2%}, Train AH: {np.mean(t_ah):.2%}")
    print(f"Gen BS: {np.mean(g_bs):.2%}, Train BS: {np.mean(t_bs):.2%}")
    print(f"Sim NW: {np.mean(nw):.2%}, SW: {np.mean(sw):.2%}")


def main():
    """Test the last three visualization/statistics functions."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test visualization and statistics functions')
    parser.add_argument('--test-voxel', action='store_true', help='Test visualize_voxel_3d_rgb')
    parser.add_argument('--test-distribution', action='store_true', help='Test visualize_data_distribution')
    parser.add_argument('--test-stat', action='store_true', help='Test stat_property')
    parser.add_argument('--pdb-path', type=str, default='./pdb/pdbs/1c3a_relaxed_rank_1_model_3.pdb',
                        help='Path to PDB file for voxel visualization')
    parser.add_argument('--output-voxel', type=str, default='test_voxel_3d.jpg',
                        help='Output path for voxel visualization')
    parser.add_argument('--gen-path', type=str, default='/cbica/home/gongha/comp_space/codes/M3CAD/identification/v4_filter_anti.csv',
                        help='Path to generated sequences CSV for stat_property')
    parser.add_argument('--train-path', type=str, default='metadata/data_processed.csv',
                        help='Path to training data CSV for stat_property')
    parser.add_argument('--test-all', action='store_true', help='Test all three functions')
    
    args = parser.parse_args()
    
    # If no specific test is selected, test all
    if not any([args.test_voxel, args.test_distribution, args.test_stat, args.test_all]):
        args.test_all = True
    
    if args.test_all:
        args.test_voxel = True
        args.test_distribution = True
        args.test_stat = True
    
    print("=" * 60)
    print("Testing visualization and statistics functions")
    print("=" * 60)
    
    # Test 1: visualize_voxel_3d_rgb
    if args.test_voxel:
        print("\n[1/3] Testing visualize_voxel_3d_rgb...")
        try:
            # Try to find a valid PDB file
            pdb_path = args.pdb_path
            if not os.path.exists(pdb_path):
                # Try alternative paths
                alt_paths = [
                    './pdb/pdb_dbassp/example_real.pdb',
                    './pdb/pdbs/example_relaxed_rank_1_model_3.pdb'
                ]
                pdb_path = next((p for p in alt_paths if os.path.exists(p)), None)
            
            if pdb_path and os.path.exists(pdb_path):
                print(f"  Using PDB file: {pdb_path}")
                visualize_voxel_3d_rgb(pdb_path, args.output_voxel)
                print(f"  ✓ Successfully created visualization: {args.output_voxel}")
            else:
                print(f"  ⚠ PDB file not found. Skipping voxel visualization.")
                print(f"  Expected path: {args.pdb_path}")
        except Exception as e:
            print(f"  ✗ Error in visualize_voxel_3d_rgb: {e}")
    
    # Test 2: visualize_data_distribution
    if args.test_distribution:
        print("\n[2/3] Testing visualize_data_distribution...")
        try:
            visualize_data_distribution()
            print("  ✓ Successfully created class distribution plot: class_distribution.jpg")
        except Exception as e:
            print(f"  ✗ Error in visualize_data_distribution: {e}")
    
    # Test 3: stat_property
    if args.test_stat:
        print("\n[3/3] Testing stat_property...")
        try:
            if os.path.exists(args.gen_path) and os.path.exists(args.train_path):
                print(f"  Using generated sequences: {args.gen_path}")
                print(f"  Using training data: {args.train_path}")
                stat_property(args.gen_path, args.train_path)
                print("  ✓ Successfully completed property statistics")
                print("  Output files: ss_comp.jpg, sim_dist.jpg")
            else:
                print(f"  ⚠ Required files not found. Skipping stat_property.")
                if not os.path.exists(args.gen_path):
                    print(f"    Missing: {args.gen_path}")
                if not os.path.exists(args.train_path):
                    print(f"    Missing: {args.train_path}")
        except Exception as e:
            print(f"  ✗ Error in stat_property: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Testing completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()