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
   virtualenv --no-download ENV_ADDRESS
   source ENV_ADDRESS/bin/activate
   pip install --no-index --upgrade pip # upgrade pip if necessary
   pip install -r requirements.txt
   ```

## Dependencies

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
