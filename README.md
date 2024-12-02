<p align="center">
  <img src="https://github.com/bcgsc/tAMPer/blob/master/imgs/logo.png"/>
</p>

# Structure-aware deep learning model for peptides toxicity prediction

## Table of Contents

- [About](#about)
- [Files](#files)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Inputs](#inputs)
- [Usage](#usage)
- [Acknowledgement](#acknowledgement)
- [Citation](#Citation)

## About

tAMPer is a multi-modal deep learning model that predicts peptide toxicity by integrating the underlying amino acid sequence composition and the predicted three-dimensional (3D) structure. tAMPer adopts a graph-based representation for peptides, encoding their ColabFold-predicted structures. The model extracts structural features using graph neural networks, and employs recurrent neural networks to capture sequential dependencies. The self-attention mechanism is utilized to integrate features from both modalities and weighs the contribution of each amino acid in predicting toxicity.

<p align="center">
  <img src="https://github.com/bcgsc/tAMPer/blob/master/imgs/tAMPer.png" />
</p>

## Files

The project contains the following files and directories:

- `checkpoints/`: Directory to store trained model checkpoints.
- `data/`: Directory for storing datasets.
- `logs/`: Directory for storing log files.
- `results/`: Directory for storing prediction results.
- `src/GConvs.py`: Implementation of graph convolutional layers.
- `src/embeddings.py`: Generating amino acids embeddings.
- `src/aminoacids.py`: Amino acid-related utility functions.
- `src/dataset.py`: Dataset loading and preprocessing.
- `src/peptideGraph.py`: Creating graph representations of peptides.
- `src/predict_tAMPer.py`: Script for making toxicity predictions on new data.
- `src/tAMPer.py`: Main implementation of the tAMPer model.
- `src/train_tAMPer.py`: Training pipeline.
- `src/utils.py`: General utility functions.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/bcgsc/tAMPer.git
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

3. Create a python virtual environment (make sure python3 is available/loaded):
   ```bash
   pip install --upgrade pip # upgrade pip if necessary
   pip install virtualenv
   virtualenv ENV_ADDRESS
   source ENV_ADDRESS/bin/activate
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


## Inputs

Provide the peptide sequences in the FASTA format (.faa). If a peptide has C-terminal amidation, add `_AMD` at the end of the peptide name.

#### Example FASTA Format

```plaintext
>peptide_1
MKALIKLPGNRVNGFGRIGR
>peptide_2_AMD
ALWKTLLKKVLKAAA
>peptide_3
GRRPLLLRAR
```

### 3D structures

Provide the directory where the ouput of the ColabFold (`.result.zip` files) is stored. To run ColabFold, please refer to https://github.com/sokrypton/ColabFold. The name of each file should match to its correspoding sequence in the fasta file. Also, add `zip_results` by ticking its corresponding box which is located within the advanced settings section of AlphaFold2.  

If you are using localcolabfold (https://github.com/YoshitakaMo/localcolabfold) for structure predictions, please ensure to include the `--amber` flag for structure refinement (relaxation / energy minimization) and `--zip` flag which stores the results in a zip file (`.result.zip`) in order to utilize tAMPer. 

```
structures
├── peptide_1.result.zip
├── peptide_2_AMD.result.zip
├── peptide_3.result.zip
...

```

## Usage

```
PROGRAM: train_tAMPer.py & predict_tAMPer.py

USAGE(S): 
   
   ######## TRAIN ##########

   usage: train_tAMPer.py [-h] -tr_pos TR_POS -tr_neg TR_NEG -tr_pdb TR_PDB -val_pos VAL_POS -val_neg VAL_NEG -val_pdb VAL_PDB [-lr LR] [-hdim HDIM] [-gru_layers GRU_LAYERS] [-embedding_model EMBEDDING_MODEL]
                [-modality MODALITY] [-gnn_layers GNN_LAYERS] [-batch_size BATCH_SIZE] [-n_epochs N_EPOCHS] [-gard_acc GARD_ACC] [-weight_decay WEIGHT_DECAY] [-d_max D_MAX] [-lammy LAMMY]
                [-monitor MONITOR] [-pre_chkpnt PRE_CHKPNT] [-chkpnt CHKPNT] [-log LOG]


   train_tAMPer.py script runs tAMPer for training.

   options:
		-h, --help            show this help message and exit
		-tr_pos TR_POS        training toxic sequences fasta file (.faa)
		-tr_neg TR_NEG        training non-toxic sequences fasta file (.faa)
		-tr_pdb TR_PDB        address directory of train structures
		-val_pos VAL_POS      validation toxic sequences fasta file (.faa)
		-val_neg VAL_NEG      validation non-toxic sequences fasta file (.faa)
		-val_pdb VAL_PDB      address directory of val structures
		-lr LR                learning rate
		-hdim HDIM            hidden dimension of model for h_seq and h_strct
		-gru_layers GRU_LAYERS
		                    number of GRU Layers
		-embedding_model EMBEDDING_MODEL
		                    different variant of ESM2 embeddings: {t6, t12, t30, t33, t36, t48}
		-modality MODALITY    Used modality
		-gnn_layers GNN_LAYERS
		                    number of GNNs Layers
		-batch_size BATCH_SIZE
		                    batch size
		-n_epochs N_EPOCHS    max number of epochs
		-gard_acc GARD_ACC    gradient accumulation steps
		-weight_decay WEIGHT_DECAY
		                    weight decay
		-d_max D_MAX          max distance to consider two connect two residues in the graph
		-lammy LAMMY          lammy in the objective function
		-monitor MONITOR      the metric to monitor for early stopping during training
		-pre_chkpnt PRE_CHKPNT
		                    address of pre-trained GNNs
		-chkpnt CHKPNT        address to where trained model be stored
		-log LOG              address to where log file be stored


   ######## PREDICT ##########

   usage: predict_tAMPer.py [-h] -seqs SEQS -pdbs PDBS [-hdim HDIM] [-embedding_model EMBEDDING_MODEL] [-d_max D_MAX] [-chkpnt CHKPNT] [-out OUT]

   predict_tAMPer.py script runs tAMPer for prediction.

   options:
		-h, --help            show this help message and exit
		-seqs SEQS            sequences fasta file for prediction (.fasta)
		-pdbs PDBS            address directory of train structures
		-hdim HDIM            hidden dimension of model for h_seq and h_strct
		-embedding_model EMBEDDING_MODEL
		                    different variant of ESM2 embeddings: {t6, t12}
		-d_max D_MAX          max distance to consider two connect two residues in the graph
		-chkpnt CHKPNT        address of .pt checkpoint to load the model
		-out OUT
		                    address of output folder
                                                                             
EXAMPLE(S):

	train_tAMPer.py -tr_pos ../tAMPer/data/sequences/tr_pos.faa \
		-tr_neg ../tAMPer/data/sequences/tr_pos.faa \
		-tr_pdb ../tAMPer/data/tr_structures/ \
		-val_pos ../tAMPer/data/sequences/tr_pos.faa \
		-val_neg ../tAMPer/data/sequences/tr_pos.faa \
		-val_pdb ../tAMPer/data/val_structures/ \
		-pre_chkpnt ../tAMPer/checkpoints/trained/pre_GNNs.pt \
		-lr 0.0004 \
		-hdim 64 \
		-gru_layers 1 \
		-gnn_layers 1 \
		-d_max 12 \
		-embedding_model t12 \
		-batch_size 32 \
		-n_epochs 100 \
		-gard_acc 1 \
		-weight_decay 1e-7 \
		-lammy 0.2 \
		-chkpnt ../tAMPer/checkpoints/chkpnt.pt \
		-log ../tAMPer/logs/log.npy
      
	predict_tAMPer.py -seqs ../data/sequences/seqs.faa \
		-pdbs ../tAMPer/data/structures/ \
		-hdim 64 \
		-embedding_model t12 \
		-d_max 12 \
		-chkpnt ../tAMPer/checkpoints/trained/chkpnt.pt \
		-out ../tAMPer/results/prediction.csv
      
```

## Acknowledgement

The implementation of portions of the GNNs convolutional layers and the input data pipeline were adapted from [Jing et al, ICLR 2021](https://github.com/drorlab/gvp) and [Baldassarre et al, Structural bioinformatics 2021](https://github.com/baldassarreFe/graphqa).

## Citation

If you use tAMPer in your research, please cite:

Ebrahimikondori H, Sutherland D, Yanai A, Richter A, Salehi A, Li C, Coombe L, Kotkoff M, Warren RL, Birol I. 2024. Structure-aware deep learning model for peptide toxicity prediction. Protein Science. https://doi.org/10.1002/pro.5076.
