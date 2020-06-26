# tAMPer: Predicting toxicity of antimicrobial peptides.

This repository contains code for running our model, with both the possibility to train your own sequences, or predict for your purposes.

The very first step is to visit [this github page](https://github.com/mheinzinger/SeqVec), or directly download the SeqVec trained on UniRef50, available at [SeqVec-model](https://rostlab.org/~deepppi/seqvec.zip).

After downloading this file, you should decompress and ideally place them under your working directory in a new folder called `seqvec`.

# Requirements

* `Python >=3.6.7`
* `torch >=0.4.1`
* `allenlp`
* `scikit-learn`

# Usage

For performing a fair comparison of your method against ours, you would need to train our model against your data. For that purpose you can use the **train** mode.
In order to predict the results of a pre-trained model on your test sequences, you should use the **predict** mode.

## **train**

When you run the `main.py` script, you should write the `train` positional argument:

``` python3 main.py train [options] ```

Parameters/options used for training are:
- `--pos`: Positive training data fasta file.
- `--neg`: Negative training data fasta file.
- `--name`: A name for the model to be trained.
- `--seqvec_path`: Path to trained seqvec model (.../uniref50_v2). Add file directory of where your SeqVec model lies.

The `train` mode will create a new `models/<name>/` directory if one does not already exist, and it will store the models under that directory.
Note: If you want to have different top level classifier(s) for your purposes, modify the `models.py` `base_models()` function.

## **predict**

When you run the `main.py` script, you should write the `predict` positional argument:

``` python3 main.py predict [options] ```

Parameters/options used for predicting are:

- `--sequences`: Custom fasta file for prediction using a saved model.
- `--models_dir`: Name of the directory under which models to be ensembled for prediction are stored. Starts with models/.
- `--out`: Name of the file with which to save predictions.
- `--seqvec_path`: [common with train mode] Path to trained seqvec model (.../uniref50_v2). Add file directory of where your SeqVec model lies.
