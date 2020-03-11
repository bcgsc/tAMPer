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

When you run the `main.py` script, you should write the `train` positional argument, like so:

``` python3 main.py train [options] ```

Parameters/options used for training are:
- `--pos`: fasta file containing positive training samples.
- `--neg`: fasta file containing negative training samples.
- `--name`: a name for your new trained model, e.g. `funnytool_tamper_model` if you are developing *funnytool* and using the default `tAMPer` model (`SeqVec` + `MLP`).
- `--custom_model`: If you want to have another top level classifier instead of the default `MLP` then you should add a path to a file that contains a function `get_model` that returns such a model. Make sure the model has a `train` and `predict` function, or make necessary modifications to these scripts for handling classification using your custom model.
- `--seqvec_path`: If you have added the downloaded SeqVec model under `seqvec/` repo, you should not worry about adding a path here, o/w add your trained SeqVec model path.

## **predict**

When you run the `main.py` script, you should write the `predict` positional argument, like so:

``` python3 main.py predict [options] ```

Parameters/options used for predicting are:

- `--sequences`: fasta file containing the sequences you would like to test.
- `--model_name`: the name of the model that has been trained by the `train` mode. Default is already pretrained `tAMPer` model.

## Visualizations

This includes visualizations that accompany the paper.