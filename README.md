![Project Logo](https://github.com/bcgsc/tAMPer/blob/master/img/logo.png) <!-- If you have a project logo, include it here -->

# tAMPer: Structure-Aware Deep Learning Model for Toxicity Prediction of Antimicrobial Peptides

## Table of Contents

- [About](#about)
- [Files](#files)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## About

tAMPer is a deep learning model designed for predicting the toxicity of antimicrobial peptides (AMPs) by considering their sequential and structural features. This model aims to enhance the accuracy of toxicity predictions for AMPs, which play a crucial role in the defense mechanisms of various organisms.

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
3. Create a conda environment (optional but recommended):
   ```bash
   conda env create -f environment.yml
   conda activate tAMPer
   ```
