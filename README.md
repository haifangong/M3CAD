# De novo multi-mechanism antimicrobial peptide design via multimodal deep learning

## Overview

Artificial Intelligence (AI)-driven discovery of antimicrobial peptides (AMPs) has significant potential in combating multidrug-resistant organisms (MDROs). However, existing approaches often overlook the three-dimensional (3D) structural characteristics, species-specific antimicrobial activities, and underlying mechanisms of AMPs.

**M3-CAD** (Multimodal, Multitask, Multilabel, and Conditionally Controlled AMP Discovery) is a cutting-edge pipeline designed to address these challenges. Leveraging the comprehensive **QLAPD** database, which comprises the sequences, structures, and antimicrobial properties of 12,914 AMPs, M3-CAD facilitates the de novo design of multi-mechanism AMPs with enhanced efficacy and reduced toxicity.
M3CAD is a deep learning framework for the design and identification of Antimicrobial Peptides (AMPs). It leverages multimodal learning by combining 1D sequence information with 3D structural (voxel) data to improve the generation of novel peptides and the prediction of their properties.

The repository is structured into two main modules:
1.  **Generation**: A Generative Model (VAE/WAE) to design novel AMP sequences conditioned on specific properties.
2.  **Identification**: A Classification/Regression Model to predict properties (e.g., antimicrobial activity, toxicity) of peptide sequences.

---

## 🚀 Getting Started

By integrating these modules, M3-CAD significantly enhances the structural characterization and functional prediction of AMPs, facilitating the discovery of candidates like **QL-AMP-1**, which exhibits four antimicrobial mechanisms with low toxicity and high efficacy against MDROs.

### Requirements

All project dependencies are specified in the `requirement.yml` file. To create the Conda environment, execute the following command:

```
conda env create -f requirement.yml
conda activate torch
```

## Usage

### Generation

To train the generation model and generate novel AMP sequences:

1. **Training the Generation Model**

   Navigate to the `generation` folder and execute the `train.sh` script

2. **Generating Sequences**

Once training is complete, use the `sample.sh` script to generate sequences

## 🧬 Generation Module

The **Generation** module focuses on creating new peptide sequences. It supports Variational Autoencoders (VAE) and Wasserstein Autoencoders (WAE) with support for both sequence-only and multimodal (sequence + voxel) inputs.

### Directory Structure (`generation/`)
- `train.py`: Main script for training the generative models.
- `eval.py`: Script for generating sequences using trained models.
- `model.py`: Definitions of model architectures (`SEQVAE`, `MMVAE`, `SEQWAE`, `MMWAE`).
- `dataset.py`: Data loading utilities for generation tasks.
- `inference.ipynb`: Jupyter Notebook for interactive generation and analysis.

### Training

To train a new generative model, use `train.py`.

**Example: Train a Sequence-based VAE**
```bash
cd generation
python train.py --gen_model vae --model seq --epoch 200
```

**Arguments:**
- `--gen_model`: Type of generative model (`vae` or `wae`).
- `--model`: Architecture type (`seq` for sequence-only, `mm_unet` or `mm_mt` for multimodal).
- `--epoch`: Number of training epochs.

### Inference / Generation

To generate new peptides using a pre-trained model:

**Option 1: Command Line**
```bash
cd generation
python eval.py --gen_model vae --model seq --weight_path runs/checkpoints/seq1/weights/best.pth
```

**Option 2: Interactive Notebook**
Open `generation/inference.ipynb` in Jupyter. This notebook allows you to:
- Load a trained model.
- Define target conditions (e.g., specific property values).
- Generate sequences and visualize the latent space.

---

## 🔍 Identification Module

The **Identification** module predicts the properties of peptides, such as whether they are antimicrobial or toxic. It uses 3D ResNets for structural data and RNNs/Transformers for sequence data.

### Directory Structure (`identification/`)
- `train.py`: Script to train classification or regression models.
- `inference.py`: Script for running predictions on new datasets.
- `network.py`: Definitions of prediction models (`MMPeptide`, `VoxPeptide`, `ResNet3D`).
- `dataset.py`: Data loading for identification tasks.
- `inference.ipynb`: Jupyter Notebook for interactive prediction.

### Training

To train an identification model (e.g., for antimicrobial activity prediction):

```bash
cd identification
python train.py --task anti --epoch 100
```

**Arguments:**
- `--task`: The target property to predict (e.g., `anti` for antimicrobial, `tox` for toxicity).
- `--epoch`: Number of training epochs.
- *(Check `train.py` arguments for more options like learning rate, batch size, etc.)*

### Inference

To predict properties for a list of sequences:

**Option 1: Interactive Notebook**
Open `identification/inference.ipynb`. This notebook demonstrates how to:
- Load the model and tokenizer.
- Input a list of peptide sequences.
- Obtain prediction scores (probabilities or regression values).

**Option 2: Script**
(If available) Use `inference.py` to process a CSV or FASTA file of sequences.

---

## 📂 Project Layout

```
M3CAD/
├── README.md            # This file
├── generation/          # Code for designing novel AMPs
│   ├── model.py         # VAE/WAE architectures
│   ├── train.py         # Training loop for generation
│   ├── eval.py          # Generation/Evaluation script
│   └── ...
└── identification/      # Code for property prediction
    ├── network.py       # Classification/Regression architectures
    ├── train.py         # Training loop for identification
    ├── inference.py     # Inference script
    └── ...
```


## Checkpoints

Pre-trained model checkpoints are available for download to facilitate quick experimentation and deployment. Access them via the following Baidu Drive link:

- **Download Link:** [Baidu Drive](https://pan.baidu.com/s/1B-7nd-av2oFi-gQtTd-Xmg?pwd=m3cd)
- **Extraction Code:** `m3cd`

*Note: A Baidu account is required to access and download the files.*

## Results

The M3-CAD pipeline has successfully designed **QLX-3DV-1** to **QLX-3DV-20** , a group of AMP with four distinct antimicrobial mechanisms. This peptide demonstrates:
- **Low Toxicity**: Minimal adverse effects on host cells.
- **High Efficacy**: Significant activity against MDROs.
- **In Vivo Validation**: In a skin wound infection model, QL-AMP-1 exhibited considerable antimicrobial effects with negligible toxicity, underscoring its therapeutic potential.

## Citation

If you find this project useful, please consider cite the following paper:
```
@article{wang2024novo,
    title={De novo multi-mechanism antimicrobial peptide design via multimodal deep learning},
    author={Wang, Yue and Gong, Haifan and Li, Xiaojuan and Li, Lixiang and Zhao, Yinuo and Bao, Peijing and Kong, Qingzhou and Wan, Boyao and Zhang, Yumeng and Zhang, Jinghui and others},
    journal={bioRxiv},
    pages={2024--01},
    year={2024},
    publisher={Cold Spring Harbor Laboratory}
}
```
