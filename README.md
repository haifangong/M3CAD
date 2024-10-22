# De novo multi-mechanism antimicrobial peptide design via multimodal deep learning

## Overview

Artificial Intelligence (AI)-driven discovery of antimicrobial peptides (AMPs) has significant potential in combating multidrug-resistant organisms (MDROs). However, existing approaches often overlook the three-dimensional (3D) structural characteristics, species-specific antimicrobial activities, and underlying mechanisms of AMPs.

**M3-CAD** (Multimodal, Multitask, Multilabel, and Conditionally Controlled AMP Discovery) is a cutting-edge pipeline designed to address these challenges. Leveraging the comprehensive **QLAPD** database, which comprises the sequences, structures, and antimicrobial properties of 12,914 AMPs, M3-CAD facilitates the de novo design of multi-mechanism AMPs with enhanced efficacy and reduced toxicity.

### Key Features

- **Multimodal Integration**: Combines sequence data with 3D structural information using an innovative 3D voxel coloring method to capture the nuanced physicochemical context of amino acids.
- **Multitask Learning**: Simultaneously addresses multiple prediction tasks, including antimicrobial activity, toxicity, and mechanism classification.
- **Multilabel Classification**: Enables the prediction of multiple antimicrobial mechanisms for each peptide.
- **Conditionally Controlled Generation**: Allows for the design of AMPs tailored to specific microbial species and desired mechanisms of action.
- **Comprehensive Database**: Utilizes the `QLAPD` database, ensuring robust training and validation of models.

## Pipeline Components

The M3-CAD pipeline integrates three primary modules:

1. **Generation Module**: Generates novel AMP sequences with desired properties.
2. **Regression Module**: Predicts continuous properties such as toxicity levels.
3. **Classification Module**: Classifies antimicrobial mechanisms and species-specific activities.

By integrating these modules, M3-CAD significantly enhances the structural characterization and functional prediction of AMPs, facilitating the discovery of candidates like **QL-AMP-1**, which exhibits four antimicrobial mechanisms with low toxicity and high efficacy against MDROs.

## Installation

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

   Navigate to the `generation` folder and execute the `train.sh` script:
   ```
   cd generation
   ./train.sh
   ```

2. **Generating Sequences**

Once training is complete, use the `sample.sh` script to generate sequences:

    ```
    ./sample.sh
    ```
### Identification

To train the identification model responsible for predicting antimicrobial properties and mechanisms:

**Training the Identification Model**

Execute the `train.sh` script located in the Identification directory.

## Checkpoints

Pre-trained model checkpoints are available for download to facilitate quick experimentation and deployment. Access them via the following Baidu Drive link:

- **Download Link:** [Baidu Drive](https://pan.baidu.com/s/1B-7nd-av2oFi-gQtTd-Xmg?pwd=m3cd)
- **Extraction Code:** `m3cd`

*Note: A Baidu account is required to access and download the files.*

## Results

The M3-CAD pipeline has successfully designed **QL-AMP-1**, an AMP with four distinct antimicrobial mechanisms. This peptide demonstrates:

- **Low Toxicity**: Minimal adverse effects on host cells.
- **High Efficacy**: Significant activity against MDROs.
- **In Vivo Validation**: In a skin wound infection model, QL-AMP-1 exhibited considerable antimicrobial effects with negligible toxicity, underscoring its therapeutic potential.

## Citation

If you find this project useful, please consider cite the following paper:

@article{wang2024novo,
    title={De novo multi-mechanism antimicrobial peptide design via multimodal deep learning},
    author={Wang, Yue and Gong, Haifan and Li, Xiaojuan and Li, Lixiang and Zhao, Yinuo and Bao, Peijing and Kong, Qingzhou and Wan, Boyao and Zhang, Yumeng and Zhang, Jinghui and others},
    journal={bioRxiv},
    pages={2024--01},
    year={2024},
    publisher={Cold Spring Harbor Laboratory}
}
