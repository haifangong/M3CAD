import os

import mdtraj as md
import numpy as np
import pandas as pd
import torch
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB import PDBParser, is_aa
from torch.utils.data import Dataset
from tqdm import tqdm
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import math


def clamp(n, smallest, largest):
    return sorted([smallest, n, largest])[1]


ATOMS = {'H': 1, 'C': 12, 'N': 14, 'O': 16, 'S': 30}
ATOMS_R = {'H': 1, 'C': 1.5, 'N': 1.5, 'O': 1.5, 'S': 2}
AMINO_ACID_WATER = {'A': 255, 'V': 255, 'P': 255, 'F': 255, 'W': 255, 'I': 255, 'L': 255, 'G': 155, 'M': 155,
                    'Y': 55, 'S': 55, 'T': 55, 'C': 55, 'N': 55, 'Q': 55, 'D': 55, 'E': 55, 'K': 55, 'R': 55, 'H': 55}
AMINO_ACID_CHARGE = {'D': 55, 'E': 55, 'A': 155, 'V': 155, 'P': 155, 'F': 155, 'W': 155, 'I': 155, 'L': 155, 'G': 155,
                     'M': 155, 'Y': 155, 'S': 155, 'T': 155, 'C': 155, 'N': 155, 'Q': 155, 'K': 255, 'R': 255, 'H': 255}


def pdb_parser(parser, idx, pdb_path):
    """

    """
    voxel = np.zeros((3, 64, 64, 64), dtype=np.int8)
    structure = parser.get_structure(idx, pdb_path)
    id = ''

    for i in structure[0]:
        id = i.id
    chain = structure[0][id]
    for res in chain:
        if is_aa(res.get_resname(), standard=True):
            resname = res.get_resname()
            amino = protein_letters_3to1[resname]
            ATOM_WATER = AMINO_ACID_WATER[amino]
            ATOM_CHARGE = AMINO_ACID_CHARGE[amino]
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
                if ATOM_R == 1:
                    voxel[0, x_i - 1:x_i, y_i - 1:y_i, z_i - 1:z_i] = ATOM_WEIGHT
                    voxel[1, x_i - 1:x_i, y_i - 1:y_i, z_i - 1:z_i] = ATOM_WATER
                    voxel[2, x_i - 1:x_i, y_i - 1:y_i, z_i - 1:z_i] = ATOM_CHARGE
                elif ATOM_R == 1.5:
                    voxel[0, x_i - 1:x_i + 1, y_i - 1:y_i + 1, z_i - 1:z_i + 1] = ATOM_WEIGHT
                    voxel[1, x_i - 1:x_i + 1, y_i - 1:y_i + 1, z_i - 1:z_i + 1] = ATOM_WATER
                    voxel[2, x_i - 1:x_i + 1, y_i - 1:y_i + 1, z_i - 1:z_i + 1] = ATOM_CHARGE
                else:
                    voxel[0, x_i - ATOM_R: x_i + ATOM_R, x_i - ATOM_R: x_i + ATOM_R,
                    x_i - ATOM_R: x_i + ATOM_R] = ATOM_WEIGHT
                    voxel[1, x_i - ATOM_R: x_i + ATOM_R, x_i - ATOM_R: x_i + ATOM_R,
                    x_i - ATOM_R: x_i + ATOM_R] = ATOM_WATER
                    voxel[2, x_i - ATOM_R: x_i + ATOM_R, x_i - ATOM_R: x_i + ATOM_R,
                    x_i - ATOM_R: x_i + ATOM_R] = ATOM_CHARGE
    return voxel


# Amino Acid to Index Mapping
AMAs = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13,
        'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, 'PAD': 0, 'END': 21}

# Index to Amino Acid Mapping
idx2ama = {1: 'A', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'P',
           14: 'Q', 15: 'R', 16: 'S', 17: 'T', 18: 'V', 19: 'W', 20: 'Y', 0: 'PAD', 21: 'END'}


def estimate_alpha_helix(pdb_file, sequence):
    traj = md.load(pdb_file)
    secondary_structure = md.compute_dssp(traj)
    alpha_helices_count = (secondary_structure[0] == 'H').sum()
    return alpha_helices_count / len(sequence)


p = PDBParser()


def calculate_property(seq):
    """
    Calculates various physicochemical properties of a protein sequence,
    including hydrophobicity and hydrophobic moment.

    Parameters:
    seq (str): Amino acid sequence.

    Returns:
    list: A list of computed properties.
    """

    # Ensure the sequence is in uppercase to match standard amino acid codes
    seq = seq.upper()

    # Initialize ProteinAnalysis object
    analysed_seq = ProteinAnalysis(seq)

    # Retrieve the sequence as a list of amino acids
    amino_acids = analysed_seq.count_amino_acids()

    # Check if the sequence is non-empty
    sequence_length = len(seq)
    if sequence_length == 0:
        raise ValueError("The input sequence is empty.")

    # -------------------------------
    # 1. GRAVY Score
    # GRAVY (Grand Average of Hydropathy) indicates the overall hydrophobicity
    # GRAVY = (Hydropathy values of all amino acids) / Number of residues
    # Biopython's gravy() already provides this value
    gravy_score = round(analysed_seq.gravy(), 3)  # Scaling by 10 as per original code

    # -------------------------------
    # 2. Aliphatic Index
    # Aliphatic Index is a measure of the relative volume occupied by aliphatic side chains
    # It is calculated as: (A + 2.9V + 3.9(I + L)) / Total Residues
    # While there is no direct function, we can compute it
    aliphatic_components = (
            amino_acids.get('A', 0) +
            2.9 * amino_acids.get('V', 0) +
            3.9 * (amino_acids.get('I', 0) + amino_acids.get('L', 0))
    )
    aliphatic_index = (aliphatic_components / sequence_length)  # Scaling by 10
    aliphatic_index = round(aliphatic_index, 3)

    # -------------------------------
    # 3. Aromaticity
    # Aromaticity is the relative frequency of aromatic amino acids
    # Biopython's aromaticity() computes this value
    aromaticity = round(analysed_seq.aromaticity(), 3)  # Scaling by 10

    # -------------------------------
    # 4. Instability Index
    # Predicts the stability of the protein in a test tube
    # A value below 40 indicates stability
    instability_index = round(analysed_seq.instability_index(), 3)

    # -------------------------------
    # 5. Secondary Structure Fractions
    # Retrieves the fractions of the sequence in alpha-helix, beta-sheet, and turn
    alpha_helix, beta_helix, turn = analysed_seq.secondary_structure_fraction()
    alpha_helix_frac = round(alpha_helix, 3)  # Scaling by 10
    beta_helix_frac = round(beta_helix, 3)  # Scaling by 10
    turn_frac = round(turn, 3)  # Scaling by 10

    # -------------------------------
    # 6. Charge at pH 7
    # Calculates the net charge of the protein at a specified pH
    # Using Biopython's charge_at_pH function
    charge_at_pH7 = round(analysed_seq.charge_at_pH(7.0), 3)

    # -------------------------------
    # 7. Isoelectric Point
    # pH at which the protein has no net charge
    isoelectric_point = round(analysed_seq.isoelectric_point(), 3)

    # -------------------------------
    # 8. Charge Density
    # Charge Density is the net charge divided by the sequence length
    charge_density = round(charge_at_pH7 / sequence_length, 3)  # Scaling by 10

    # -------------------------------
    # 9. Average Hydrophobicity
    # Using Kyte-Doolittle hydrophobicity scale
    hydrophobicity_scale = {
        'A': 1.8,
        'R': -4.5,
        'N': -3.5,
        'D': -3.5,
        'C': 2.5,
        'Q': -3.5,
        'E': -3.5,
        'G': -0.4,
        'H': -3.2,
        'I': 4.5,
        'L': 3.8,
        'K': -3.9,
        'M': 1.9,
        'F': 2.8,
        'P': -1.6,
        'S': -0.8,
        'T': -0.7,
        'W': -0.9,
        'Y': -1.3,
        'V': 4.2
    }

    hydrophobic_sum = 0
    valid_residues = 0
    for aa in seq:
        if aa in hydrophobicity_scale:
            hydrophobic_sum += hydrophobicity_scale[aa]
            valid_residues += 1
        else:
            # Non-standard amino acids are ignored
            pass

    average_hydrophobicity = round((hydrophobic_sum / valid_residues) if valid_residues > 0 else 0, 3)

    # -------------------------------
    # 10. Hydrophobic Moment
    # Hydrophobic Moment measures the amphipathicity of a helix
    # It is calculated using the vectors of hydrophobicities multiplied by their helical wheel angles
    # Assuming an alpha-helix with 3.6 residues per turn

    # Parameters for alpha-helix
    residues_per_turn = 3.6
    angle_increment = (360 / residues_per_turn)  # degrees per residue

    hydrophobic_moment_x = 0
    hydrophobic_moment_y = 0

    for i, aa in enumerate(seq):
        if aa in hydrophobicity_scale:
            hydrophobicity = hydrophobicity_scale[aa]
            angle_deg = i * angle_increment
            angle_rad = math.radians(angle_deg % 360)  # Convert to radians
            hydrophobic_moment_x += hydrophobicity * math.cos(angle_rad)
            hydrophobic_moment_y += hydrophobicity * math.sin(angle_rad)
        else:
            # Non-standard amino acids contribute nothing to hydrophobic moment
            pass

    hydrophobic_moment = round(
        (math.sqrt(
            hydrophobic_moment_x ** 2 + hydrophobic_moment_y ** 2) / valid_residues) if valid_residues > 0 else 0,
        3
    )

    # -------------------------------
    # Compile all computed properties into a list
    properties = [
        gravy_score,  # GRAVY Score
        aliphatic_index,  # Aliphatic Index
        aromaticity,  # Aromaticity
        instability_index,  # Instability Index
        alpha_helix_frac,  # Alpha Helix Fraction
        beta_helix_frac,  # Beta Helix Fraction
        turn_frac,  # Turn Fraction (scaled by 10)
        charge_at_pH7,  # Charge at pH 7
        isoelectric_point,  # Isoelectric Point
        charge_density,  # Charge Density
        average_hydrophobicity,  # Average Hydrophobicity
        hydrophobic_moment  # Hydrophobic Moment
    ]

    return properties


class ADataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''

    def __init__(self, mode='train', sequence_length=30):
        self.data_list = []
        all_data = pd.read_csv('./metadata/data_515.csv', encoding="unicode_escape").values#[:300]
        # all_data = pd.read_csv('./metadata/data.csv', encoding="unicode_escape").values
        # 0-13 anti 2-13 specific [763 5619 5227 1156 4504 14 1561 0 9 95 0 138]
        # 14-20 toxic
        # 21-24 machine
        print(all_data.shape)

        idx_list, seq_list, labels = all_data[:, 0], all_data[:, 1], all_data[:, 2:]
        filter_data = []

        label_list = np.concatenate((labels[:, 1:2], labels[:, -2:]), axis=1)
        label_list = labels[:, 1:2]
        for idx in range(len(idx_list)):
            seq = seq_list[idx].upper().strip()
            # gt = [1 if int(i) > 0 else 0 for i in label_list[idx]]
            gt = [1 if int(i) > 0 else 0 for i in label_list[idx]]
            filter_data.append((seq, gt, str(idx_list[idx])))
        print("Fliter data finished ", len(filter_data))
        splited_data = filter_data
        gt_list = []
        for item in tqdm(splited_data):
            seq, gt, idx = item
            if len(seq) > sequence_length - 1:
                continue
            if len(seq) < 10:
                continue
            idx = str(idx).zfill(5)

            if os.path.exists("../identification/pdb/pdbs/" + seq + "_relaxed_rank_1_model_3.pdb"):
                pdb_path = "../identification/pdb/pdbs/" + seq + "_relaxed_rank_1_model_3.pdb"
            elif os.path.exists("../identification/pdb/pdbs/" + seq + "_real.pdb"):
                pdb_path = "../identification/pdb/pdbs/" + seq + "_real.pdb"
            else:
                print(seq)
                continue
                assert os.path.exists("./pdb_gen/pdb_dbassp/" + seq + "_real.pdb")
            properties = calculate_property(seq)
            gravy_score, aliphatic_index, aromaticity, instability_index, alpha_helix_frac, beta_helix_frac, turn_frac, charge_at_pH7, isoelectric_point, charge_density, average_hydrophobicity, hydrophobic_moment = properties
            # print(alpha_helix_frac)
            alpha_helix_label = 1 if alpha_helix_frac > 0.3 else 0
            stability_label = 1 if instability_index < 35 else 0
            charge_label = 1 if charge_at_pH7 > 1 else 0

            gt.append(alpha_helix_label)
            gt.append(stability_label)
            gt.append(charge_label)
            gt_list.append(gt)
            voxel = pdb_parser(p, idx, pdb_path)

            seq_code = [AMAs[char] for char in seq] + [0]
            # seq_emb = seq_code
            seq_emb = [seq_code + [21] * (sequence_length - len(seq_code))]
            self.data_list.append((voxel, seq_emb, gt))
        print(len(gt_list))
        print(np.sum(np.array(gt_list), axis=0))

        # self.im_aug = transforms.Compose([
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomVerticalFlip(p=0.5),
        #     transforms.RandomRotation(90, expand=False, center=None),
        # ])

    def __getitem__(self, idx):
        voxel, seq, gt = self.data_list[idx]
        # voxel = self.im_aug(torch.Tensor(voxel).float())
        return voxel, torch.Tensor(seq), torch.Tensor(gt)

    def __len__(self):
        return len(self.data_list)


def estimate_alpha_helix_fraction(sequence):
    protein_analysis = ProteinAnalysis(sequence)
    helix_fraction, _, _ = protein_analysis.secondary_structure_fraction()
    return helix_fraction

# 'IRILKWLL'
# Example usage:
# alpha_helix_fraction = estimate_alpha_helix_fraction('ATFGRCRRWWAALGACRR')

# print(charge)

# ADataset(mode='train', fold=0)
