"""
Date: 2020-02-14
Author: figalit (github.com/figalit)
"""
import os
import numpy as np
import seqvec_embedder
import pickle
import models

TIMESTEPS = 100
ENCODING_DIM = 50 # TODO: Maybe have this be an input argument?
EMB_LEN = 1024

def save_base_models(classifiers, X, y, models_dir):
    clf1, clf2, clf3, stacking = models.base_models()
    for clf in [clf1,clf2,clf3]:
        clf.fit(X,y)
        fileout = "{}/{}.sav".format(models_dir,type(clf).__name__)
        pickle.dump(clf, open(fileout, 'wb'))
    # save stacking model
    stacking.fit(X, y)
    pickle.dump(stacking, open("{}/{}.sav".format(models_dir, type(stacking).__name__), 'wb'))

def embed_seqs(sequences, seqvec_path="seqvec/uniref50_v2"):
    embs = seqvec_embedder.list_embeddings(sequences)
    embs_residue = seqvec_embedder.list_embeddings_residue(sequences)
    return embs, embs_residue

def embed(pos_seqs, neg_seqs, seqvec_path="seqvec/uniref50_v2", models_name):
    pos_embeddings = seqvec_embedder.list_embeddings(pos_seqs)
    neg_embeddings = seqvec_embedder.list_embeddings(neg_seqs)    
    pos_embeddings_residue = seqvec_embedder.list_embeddings_residue(pos_seqs)
    neg_embeddings_residue = seqvec_embedder.list_embeddings_residue(neg_seqs)
    
    X = np.concatenate((np.asarray(pos_embeddings), np.asarray(neg_embeddings)))
    y = np.concatenate((np.ones(len(pos_seqs)), np.zeros(len(neg_seqs))))

    # TODO: May need to adjust this? 
    X_residue = np.concatenate((np.asarray(pos_embeddings_residue), np.asarray(neg_embeddings_residue)))

    # TODO: Store embeddings?

    # Train and save models based on embeddings
    models_dir = "models/{}".format(models_name)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    models.save_base_models(X, y, models_dir)

    model = models.lstm_model(EMB_LEN, TIMESTEPS, ENCODING_DIM)
    history = model.fit(X_residue, y, epochs=10, validation_split=0.25, batch_size=10)
    model.save("{}{}.h5".format(models_dir, "lstm_model"))

    print("Models saved in", models_dir)

def predict(embs, embs_residue, seqs, models_dir):
    # per residue predictions
    lstm_modelname = "{}{}.h5".format(models_dir, "lstm_model")
    y_pred_res_score, y_pred_res = models.predict_lstm(lstm_modelname, embs_residue)
    # base predictions
    y_pred_base = models.predict_base(models_dir, embs)
    
    predictions = np.column_stack((y_pred_res_score, y_pred_base))
    final_preds = np.average(predictions, axis=1)
    return final_preds


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