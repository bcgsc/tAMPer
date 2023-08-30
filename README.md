<p align="center">
  <img src="https://github.com/bcgsc/tAMPer/blob/master/img/logo.png" />
</p>

# tAMPer: Structure-Aware Deep Learning Model for Toxicity Prediction of Antimicrobial Peptides

## Table of Contents

- [About](#about)
- [Files](#files)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## About

tAMPer is a deep learning model designed for predicting the toxicity of antimicrobial peptides (AMPs) by considering their sequential and structural features. tAMPer adopts a graph-based representation, where each peptide is represented as a graph that encodes AlphaFold-predicted 3D structure of the peptide. Structural features are extracted using graph neural networks, while recurrent neural networks capture sequential dependencies.

## Files

The project contains the following files and directories:

- `checkpoints/`: Directory to store trained model checkpoints.
- `data/`: Directory for storing datasets.
- `GConvs.py`: Implementation of graph convolutional layers.
- `aminoacids.py`: Amino acid-related utility functions.
- `dataset.py`: Dataset loading and preprocessing.
- `environment.yml`: Environment file for reproducing the environment.
- `peptideGraph.py`: Creating graph representations of peptides.
- `predict.py`: Script for making toxicity predictions on new data.
- `run.py`: Script for running experiments and training.
- `tAMPer.py`: Main implementation of the tAMPer model.
- `train.py`: Training pipeline.
- `utils.py`: General utility functions.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/tAMPer.git 
   ```
2. Navigate to the project directory:
   ```bash
   cd tAMPer
   ```

### Conda

3. Create a conda environment (optional but recommended):
   ```bash
   conda env create -f environment.yml
   conda activate tAMPer
   ```

### Pip

3. Create a python virtual environment environment (make sure python3 is available/loaded):
   ```bash
   python3 -m venv ENV_ADDRESS
   source ENV_ADDRESS/bin/activate
   pip install --upgrade pip # upgrade pip if necessary
   pip install -r requirements.txt
   ```

## Dependencies

- [DSSP](https://ssbio.readthedocs.io/en/latest/instructions/dssp.html)
- python (>= 3.9)
- [pytorch](https://pytorch.org/get-started/previous-versions/) (>= 1.13.1)
- [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
- torch-cluster
- torch-scatter
- [fair-esm](https://github.com/facebookresearch/esm) (>= 2.0.0)
- numpy
- pandas
- biopython
- wandb (optional)
- loguru

## Usage

```
PROGRAM: train.py & predict.py

USAGE(S): 
   
   ######## TRAIN ##########

   usage: train.py [-h] -tr_pos TR_POS -tr_neg TR_NEG -pdb_dir PDB_DIR -val_pos VAL_POS
    -val_neg VAL_NEG [-lr LR] [-hdim HDIM] [-sn SN] [-emd EMD] [-gl GL] [-bz BZ] [-eph EPH]
     [-acg ACG] [-wd WD] [-dm DM] [-lambda LAMBDA] [-monitor MONITOR] [-pck PCK] [-ck CK]
     [-log LOG]

   train.py script runs tAMPer for training.

   options:
      -h, --help        show this help message and exit
      -tr_pos TR_POS    training toxic sequences fasta file (.fasta)
      -tr_neg TR_NEG    training non-toxic sequences fasta file (.fasta)
      -pdb_dir PDB_DIR  address directory of structures
      -val_pos VAL_POS  validation toxic sequences fasta file (.fasta)
      -val_neg VAL_NEG  validation non-toxic sequences fasta file (.fasta)
      -lr LR            learning rate
      -hdim HDIM        hidden dimension of model for h_seq and h_strct
      -sn SN            number of GRU Layers
      -emd EMD          different variant of ESM2 embeddings: {t6, t12, t30, t33, t36, t48}
      -gl GL            number of GNNs Layers
      -bz BZ            batch size
      -eph EPH          max number of epochs
      -acg ACG          gradient accumulation steps
      -wd WD            weight decay of optimizer
      -dm DM            max distance to consider two connected residues in the graph
      -lambda LAMBDA    lambda in the objective function
      -monitor MONITOR  the metric to monitor for early stopping during training
      -pck PCK          address of pre-trained GNNs (.pt)
      -ck CK            address of best performing model on validation to be stored (.pt)
      -log LOG          address of log file (measured training & validation metrics for each epoch) to be stored (.npy)

   ######## PREDICT ##########

   usage: predict.py [-h] -seq SEQ -pdb_dir PDB_DIR [-dm DM] [-ck CK] [-res RES]

   predict.py script runs tAMPer for prediction.

   options:
      -h, --help        show this help message and exit
      -seq SEQ          address of sequences (.faa)
      -pdb_dir PDB_DIR  address directory of structures
      -dm DM            max distance to consider two connected residues in the graph
      -ck CK            address of checkpoint to load the model (.pt)
      -res RES          address of results to be saved (.npy)

                                                                              
EXAMPLE(S):

      python3 train.py -tr_pos ../tAMPer/data/sequences/tr_pos.faa -tr_neg ../tAMPer/data/sequences/tr_pos.faa \
      -val_pos ../tAMPer/data/sequences/tr_pos.faa -val_neg ../tAMPer/data/sequences/tr_pos.faa \
      -pdb_dir ../tAMPer/data/structures/ -pck ../tAMPer/checkpoints/trained/pre_GNNs.pt \
      -lr 0.0004 -hdim 64 -sn 1 -gl 1 -dm 12 -emd t30 -bz 32 -eph 100 -acg 2 -wd 1e-7 -lambda 0.2 \
      -ck ../tAMPer/checkpoints/chkpnt.pt -log ../tAMPer/logs/log.npy
      

      python3 predict.py -seq ../tAMPer/data/test_seq.faa -pdb_dir ../tAMPer/data/structures/ \
      -dm 12 -ck ../tAMPer/checkpoints/trained/chkpnt.pt -res ../tAMPer/results/prediction.csv
      
```
