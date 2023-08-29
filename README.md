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

   usage: train.py [-h] -tr_pos TR_POS -tr_neg TR_NEG -pdb_dir PDB_DIR -val_pos VAL_POS -val_neg VAL_NEG [-lr LEARNING_RATE] [-hdim HDIM] [-sn SEQUENCE_NUM_LAYERS] [-emd EMBEDDING] [-gl GNN_LAYERS] [-bz BATCH_SIZE] [-eph NUM_EPOCH] [-acg ACCUM_ITER] [-wd WEIGHT_DECAY] [-dm D_MAX] [-beta BETA]
                [-monitor MONITOR_METER] [-pck PRE_CHKPNT] [-ck CHECKPOINT_DIR] [-log LOG_FILE]

   train.py script runs tAMPer for training.

   options:
     -h, --help            show this help message and exit
     -tr_pos TR_POS, --tr_pos TR_POS
                           train toxic sequences address (.faa)
     -tr_neg TR_NEG, --tr_neg TR_NEG
                           traib non-toxic sequences address (.faa)
     -pdb_dir PDB_DIR, --pdb_dir PDB_DIR
                           directory of structures
     -val_pos VAL_POS, --val_pos VAL_POS
                           validation toxic sequences address (.faa)
     -val_neg VAL_NEG, --val_neg VAL_NEG
                           validation non-toxic sequences address (.faa)
     -lr LEARNING_RATE, --learning_rate LEARNING_RATE
     -hdim HDIM, --hdim HDIM
                           hidden dimension of model for h_seq and h_strct (int)
     -sn SEQUENCE_NUM_LAYERS, --sequence_num_layers SEQUENCE_NUM_LAYERS
                           number of GRU Layers (int)
     -emd EMBEDDING, --embedding EMBEDDING
                           different variant of ESM2 embeddings: {t6, t12, t30, t33, t36, t48}
     -gl GNN_LAYERS, --gnn_layers GNN_LAYERS
                           number of GNNs Layers (int)
     -bz BATCH_SIZE, --batch_size BATCH_SIZE
     -eph NUM_EPOCH, --num_epoch NUM_EPOCH
     -acg ACCUM_ITER, --accum_iter ACCUM_ITER
     -wd WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
     -dm D_MAX, --d_max D_MAX
                           max distance to consider two connect two residues in the graph (int)
     -beta BETA, --beta BETA
                           lambda in the objective function
     -monitor MONITOR_METER, --monitor_meter MONITOR_METER
                           the metric to monitor for early stopping during training
     -pck PRE_CHKPNT, --pre_chkpnt PRE_CHKPNT
                           address of pre-trained GNNs (.pt)
     -ck CHECKPOINT_DIR, --checkpoint_dir CHECKPOINT_DIR
                           address to where trained model be stored (.pt)
     -log LOG_FILE, --log_file LOG_FILE
                           address to where log file be stored (.npy)

   usage: predict.py [-h] -seq SEQUENCES -pdb_dir PDB_DIR [-dm D_MAX] [-ck CHECKPOINT_DIR] [-res RESULT_DIR]

   predict.py script runs tAMPer for prediction.

   options:
     -h, --help            show this help message and exit
     -seq SEQUENCES, --sequences SEQUENCES
                           address of sequences (.faa)
     -pdb_dir PDB_DIR, --pdb_dir PDB_DIR
                           address directory of structures
     -dm D_MAX, --d_max D_MAX
                           max distance to consider two connect two residues in the graph
     -ck CHECKPOINT_DIR, --checkpoint_dir CHECKPOINT_DIR
                           address of .pt checkpoint to load the model
     -res RESULT_DIR, --result_dir RESULT_DIR
                           address of results (.csv) to be saved

                                                                              
EXAMPLE(S):
      python3 train.py

      python3 predict.py
      
```
