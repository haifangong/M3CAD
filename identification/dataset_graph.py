import os
import pandas as pd
import numpy as np
import networkx as nx
import torch
from tqdm import tqdm
from Bio.PDB import PDBParser, is_aa
from Bio.PDB.Polypeptide import protein_letters_3to1
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# Amino Acid Mapping
AMAs = {'G': 20, 'A': 1, 'V': 2, 'L': 3, 'I': 4, 'P': 5, 'F': 6, 'Y': 7, 'W': 8, 'S': 9, 'T': 10, 'C': 11,
        'M': 12, 'N': 13, 'Q': 14, 'D': 15, 'E': 16, 'K': 17, 'R': 18, 'H': 19, 'X': 21}

class PairData(Data):
    """Custom Data object to handle graph structure and ground truth."""
    def __init__(self, edge_index_s=None, x_s=None, gt=None):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.gt = gt

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        return super().__inc__(key, value, *args, **kwargs)

def load_aa_features(feature_path):
    """Loads physical/chemical features for amino acids from a text file."""
    aa_features = {}
    if not os.path.exists(feature_path):
        return {k: [0.0]*20 for k in AMAs.keys()} # Fallback if file missing
    with open(feature_path) as f:
        for line in f:
            parts = line.strip().split()
            aa_features[parts[0]] = [float(x) for x in parts[1:]]
    return aa_features

def load_dataset(task='anti'):
    """
    Loads metadata, filters sequences, parses PDB files into graphs, 
    and returns a list of (graph, sequence_embedding, dummy, ground_truth).
    """
    data_list = []
    csv_path = 'metadata/data_processed.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file {csv_path} not found.")
    
    all_data = pd.read_csv(csv_path, encoding="unicode_escape").values
    
    # Task-specific data slicing
    if task in ['anti', 'mechanism', 'mic', 'anti-binary']:
        all_data = all_data[:12915, :]
    elif task == 'toxin':
        all_data = all_data[23666:, :]

    idx_list, seq_list, labels = all_data[:, 0], all_data[:, 1], all_data[:, 2:]
    
    # Label Processing
    if task == 'anti':
        label_list = np.concatenate((labels[:, 2:7], labels[:, 8:9]), axis=1)
    elif task == 'toxin':
        label_list = labels[:, 15:21]
    elif task == 'anti-all':
        label_list = labels[:, 2:14]
    elif task == 'mechanism':
        label_list = labels[:, 21:25]
    elif task == 'anti-binary':
        label_list = (np.sum(labels[:, [2,3,4,5,6,8]], axis=1) > 0).astype(int)
    else:
        label_list = labels

    # Filtering valid sequences
    filter_data = []
    for i in range(len(idx_list)):
        seq = str(seq_list[i]).upper().strip()
        if any(char in seq for char in 'XOU') or not (6 <= len(seq) <= 50):
            continue
        
        gt = [int(val) for val in (label_list[i] if hasattr(label_list[i], '__iter__') else [label_list[i]])]
        filter_data.append((seq, gt, str(idx_list[i])))

    # PDB Parsing and Graph Construction
    p = PDBParser(QUIET=True)
    aa_features = load_aa_features('./features.txt')

    for seq, gt, idx_raw in tqdm(filter_data, desc=f"Loading {task}"):
        # Determine PDB Path
        pdb_path = None
        paths_to_check = [
            f"./pdb/pdbs/{seq}_relaxed_rank_1_model_3.pdb",
            f"./pdb/pdbs/{seq}_real.pdb"
        ]
        for path in paths_to_check:
            if os.path.exists(path):
                pdb_path = path
                break
        
        if not pdb_path:
            continue

        try:
            structure = p.get_structure(idx_raw, pdb_path)
            # Check for valid chain A
            if 'A' not in structure[0]:
                continue
            
            chain = structure[0]['A']
            G = nx.Graph()
            
            # Add Nodes
            nodes_info = []
            for res in chain:
                if is_aa(res, standard=True):
                    res_id = res.id[1]
                    res_name_3 = res.get_resname()
                    res_name_1 = protein_letters_3to1.get(res_name_3, 'X')
                    
                    G.add_node(res_id, x=aa_features.get(res_name_1, [0.0]*20))
                    nodes_info.append(res_id)

            # Add Edges based on C-alpha distance (threshold 5 Angstroms)
            for i in range(len(nodes_info)):
                for j in range(i + 1, len(nodes_info)):
                    m, n = nodes_info[i], nodes_info[j]
                    dist = chain[m]["CA"] - chain[n]["CA"]
                    if dist <= 5:
                        G.add_edge(m, n, weight=5.0 / dist if dist > 0 else 5.0)

            if G.number_of_nodes() == 0:
                continue

            # Convert to PyG Data
            G = nx.convert_node_labels_to_integers(G)
            data_wt = from_networkx(G)
            data_graph = PairData(edge_index_s=data_wt.edge_index, x_s=data_wt.x, gt=gt)
            
            # Sequence Embedding (Padding to 30)
            seq_code = [AMAs.get(char, 21) for char in seq]
            seq_emb = [seq_code + [0] * (30 - len(seq_code))]
            
            data_list.append((data_graph, seq_emb, 0, gt))

        except Exception:
            continue

    print(f"Successfully loaded {len(data_list)} samples.")
    return data_list