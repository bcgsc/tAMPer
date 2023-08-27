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

DESCRIPTION:
      The predict.py & train.py run tAMPer for predicting and training, respectively.
      
USAGE(S): 

      Train

      python3 train.py [-tr_pos <.FASTA>] [-tr_neg <.FASTA>] [-val_pos <.FASTA>] [-val_neg <.FASTA>] [-pdb_dir <address>] [-lr <float>] [-eph <int>] [-ck <checkpoint (.pt)>] [-log <log (.npy)>] [-res <validation.csv>] [-wdb] [-sn <int>] [-emd {t6, t12, t30, t33}] []
      
OPTIONS:
       -a <address>    email address for alerts                               
       -c <class>      taxonomic class of the dataset                         (default = top-level directory in $outdir)
       -d              debug mode of Makefile                                 
       -f              force characterization even if no AMPs found           
       -h              show help menu                                         
       -m <target>     Makefile target                                        (default = exonerate)
       -n <species>    taxonomic species or name of the dataset               (default = second-level directory in $outdir)
       -o <directory>  output directory                                       (default = directory of input reads TXT file)
       -p              run processes in parallel                              
       -r <FASTA.gz>   reference transcriptome                                (accepted multiple times, *.fna.gz *.fsa_nt.gz)
       -s              strand-specific library construction                   (default = false)
       -t <int>        number of threads                                      (default = 48)
       -v              print version number                                   
       -E <e-value>    E-value threshold for homology search                  (default = 1e-5)
       -S <0 to 1>     AMPlify score threshold for amphibian AMPs             (default = 0.90)
       -L <int>        Length threshold for AMPs                              (default = 30)
       -C <int>        Charge threshold for AMPs                              (default = 2)
       -R              Disable redundancy removal during transcript assembly

       Predict

       python3 predict.py [-seq <.FASTA>] [-pdb_dir <address>] [-lr <float>] [-eph <int>] [-ck <checkpoint (.pt)>] [-log <log (.npy)>] [-res <.csv>] [-wdb]

OPTIONS:
       -a <address>    email address for alerts                               
       -c <class>      taxonomic class of the dataset                         (default = top-level directory in $outdir)
       -d              debug mode of Makefile                                 
       -f              force characterization even if no AMPs found           
       -h              show help menu                                         
       -m <target>     Makefile target                                        (default = exonerate)
       -n <species>    taxonomic species or name of the dataset               (default = second-level directory in $outdir)
       -o <directory>  output directory                                       (default = directory of input reads TXT file)
       -p              run processes in parallel                              
       -r <FASTA.gz>   reference transcriptome                                (accepted multiple times, *.fna.gz *.fsa_nt.gz)
       -s              strand-specific library construction                   (default = false)
       -t <int>        number of threads                                      (default = 48)
       -v              print version number                                   
       -E <e-value>    E-value threshold for homology search                  (default = 1e-5)
       -S <0 to 1>     AMPlify score threshold for amphibian AMPs             (default = 0.90)
       -L <int>        Length threshold for AMPs                              (default = 30)
       -C <int>        Charge threshold for AMPs                              (default = 2)
       -R              Disable redundancy removal during transcript assembly

                                                                              
EXAMPLE(S):
      python3 train.py -a user@example.com -c class -n species -p -s -t 8 -o /path/to/output/directory -r /path/to/reference.fna.gz -r /path/to/reference.fsa_nt.gz /path/to/input.txt 

      python3 predict.py -a user@example.com -c class -n species -p -s -t 8 -o /path/to/output/directory -r /path/to/reference.fna.gz -r /path/to/reference.fsa_nt.gz /path/to/input.txt 
      
INPUT EXAMPLE:
      
```
