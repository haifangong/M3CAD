import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import is_aa
from Bio.PDB.Polypeptide import protein_letters_3to1
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from scipy.ndimage import map_coordinates
from matplotlib.colors import Normalize
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
                if ATOM_R == 1:
                    voxel[0, x_i - 1:x_i, y_i - 1:y_i, z_i - 1:z_i] = ATOM_WEIGHT
                    voxel[1, x_i - 1:x_i, y_i - 1:y_i, z_i - 1:z_i] = ATOM_WATER
                    voxel[2, x_i - 1:x_i, y_i - 1:y_i, z_i - 1:z_i] = ATOM_CHARGE
                    # voxel[3, x_i - 1:x_i, y_i - 1:y_i, z_i - 1:z_i] = ATOM_CATEGORY
                elif ATOM_R <= 1.5:
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


def visualize_voxel_3d_rgb(pdb_path: str,
                           output_jpg: str,
                           iso_channel: int = 0,
                           threshold: float = 0.5,
                           figsize: tuple = (8, 8),
                           dpi: int = 300):
    """
    Parse a PDB into a (3×64×64×64) voxel grid, extract an isosurface
    on channel `iso_channel` at `threshold`, sample all three channels
    at the surface, and color each face by its (R,G,B) = (ch0,ch1,ch2).

    Parameters
    ----------
    pdb_path : str
        Path to input PDB file.
    output_jpg : str
        Path where the JPEG will be saved.
    iso_channel : int
        Which channel [0,1,2] to run marching_cubes on.
    threshold : float
        Surface level for marching_cubes.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Resolution of saved JPEG.
    """
    # 1) Parse into voxel grid
    parser = PDBParser(QUIET=True)
    voxel, _ = pdb_parser(parser, 'struct', pdb_path)
    # Three volumes: weight, water, charge
    volumes = [voxel[i].astype(np.float32) for i in range(3)]

    # 2) Extract isosurface on the chosen channel
    verts, faces, normals, vals0 = measure.marching_cubes(
        volumes[iso_channel],
        level=threshold,
        spacing=(1.0, 1.0, 1.0)
    )
    # verts: (n_verts,3), faces: (n_faces,3)

    # 3) Sample all three volumes at the surface vertices
    #    map_coordinates expects coords in order [z, y, x], but our verts
    #    are (x, y, z) so we reorder.
    coords = np.vstack([verts[:, 2], verts[:, 1], verts[:, 0]])
    sampled = np.zeros((verts.shape[0], 3), dtype=np.float32)
    for ch in range(3):
        sampled[:, ch] = map_coordinates(
            volumes[ch],
            coords,
            order=1,
            mode='nearest'
        )

    # 4) For each face, average its three corner‐vertex values per channel
    face_vals = sampled[faces].mean(axis=1)  # (n_faces, 3)

    # 5) Normalize each channel to [0,1] for RGB
    norms = [Normalize(vmin=vol.min(), vmax=vol.max()) for vol in volumes]
    face_colors = np.zeros((faces.shape[0], 4), dtype=np.float32)
    for ch in range(3):
        face_colors[:, ch] = norms[ch](face_vals[:, ch])
    face_colors[:, 3] = 1.0  # opaque

    # 6) Plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces],
                            facecolors=face_colors,
                            linewidth=0.05,
                            edgecolor='gray')
    ax.add_collection3d(mesh)

    # Axes limits & labels
    lim = volumes[0].shape[0]
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_zlim(0, lim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'RGB Isosurface (iso_ch={iso_channel}, thr={threshold})')

    # Legend for channels
    ax.text2D(0.02, 0.95,
              "R=weight (ch0), G=water (ch1), B=charge (ch2)",
              transform=ax.transAxes,
              fontsize=10)

    plt.tight_layout()
    plt.savefig(output_jpg, dpi=dpi)
    plt.close(fig)
