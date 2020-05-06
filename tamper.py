"""
Date: 2020-02-14
Author: figalit (github.com/figalit)
"""

import seqvec_embedder
DEFAULT_SEQVEC_PATH = "seqvec/uniref50_v2" # Default path for seqvec pretrained model.

def embed(pos_seqs, neg_seqs, seqvec_path=DEFAULT_SEQVEC_PATH):
    # Compute and store the seqvec embeddings in a new directory


    pos_embeddings = seqvec_embedder.get_list_embedding(pos_seqs)
    neg_embeddings = seqvec_embedder.get_list_embedding(neg_seqs)
    
    pos_embeddings_residue = seqvec_embedder.get_list_embedding_per_residue(pos_seqs)
    neg_embeddings_residue = seqvec_embedder.get_list_embedding(neg_seqs)
    



import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from allennlp.commands.elmo import ElmoEmbedder
import pickle

PREDICTIONS_DIR = '/projects/btl/ftaho/tAMPer/predictions'
WEIGHTS = 'weights.hdf5'
OPTIONS = 'options.json'

def get_seqvec(seqvec_path):
    model_dir = Path(seqvec_path)
    weights = model_dir / WEIGHTS
    options = model_dir / OPTIONS
    seqvec  = ElmoEmbedder(options, weights, cuda_device =-1)
    return seqvec

def seqs_embed(seqs, seqvec_path):
    seqvec = get_seqvec(seqvec_path)
    embedding = seqvec.embed_sentence(seqs)
    return np.asarray(torch.tensor(embedding).sum(dim=0))

def train_embed(pos_seqs, neg_seqs, seqvec_path):
    seqvec = get_seqvec(seqvec_path)
    pos_embedding = seqvec.embed_sentence(pos_seqs)
    neg_embedding = seqvec.embed_sentence(neg_seqs)
    pos_seqvec = torch.tensor(pos_embedding).sum(dim=0)
    neg_seqvec = torch.tensor(neg_embedding).sum(dim=0)
    
    pos_labels = np.ones(len(pos_seqs))
    neg_labels = np.zeros(len(neg_seqs))

    X_train = np.concatenate((np.asarray(pos_seqvec), np.asarray(neg_seqvec)))
    y_train = np.concatenate((pos_labels, neg_labels))
    return X_train, y_train

def get_model():
    from sklearn import metrics
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(1000, 500, 800),solver='adam')
    return model

def train(model, X_train, y_train, name):
    print("Training model.")
    model.fit(X_train, y_train)
    print("Model fit. Saving model.")
    filename = 'models/{}.sav'.format(name)
    pickle.dump(model, open(filename, 'wb'))

def predict(modelname, X_test, seqs):
    modelpath = 'models/{}.sav'.format(modelname)
    model = pickle.load(open(modelpath, 'rb'))
    y_pred = model.predict(X_test)
    if not os.path.exists(PREDICTIONS_DIR): os.mkdir(PREDICTIONS_DIR)
    saved_predictions_filename = '{}/{}.csv'.format(PREDICTIONS_DIR, modelname)
    f = open(saved_predictions_filename,'w+')
    f.write("sequences,predictions\n")
    for i,elem in enumerate(seqs):
        f.write("{},{}\n".format(elem,y_pred[i]))
    f.close()
    print("Predictions saved in ", saved_predictions_filename)