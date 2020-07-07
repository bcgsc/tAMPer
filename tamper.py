"""
Date: 2020-02-14
Author: figalit (github.com/figalit)
"""
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import os
import numpy as np
import pandas as pd
import seqvec_embedder
import pickle
import models

TIMESTEPS = 100
ENCODING_DIM = 50 # TODO: Potential extension, have this be an input argument.
EMB_LEN = 1024
SEQ_MODEL_NAME = "BiLSTMSequentialClassifier"

def save_base_models(X, y, models_dir):
    clf1, clf2, clf3 = models.base_models()

    for clf in [clf1,clf2,clf3]:
        clf.fit(X,y)
        fileout = "{}/{}.sav".format(models_dir,type(clf).__name__)
        pickle.dump(clf, open(fileout, 'wb'))
    
    # save stacking model
    stacking = models.stacking_fit(X, y, models_dir)
    pickle.dump(stacking, open("{}/stacking/StackingClassifier.sav".format(models_dir), 'wb'))
    print("Saved Seqvec embedding models in", models_dir)

def embed_seqs(sequences, seqvec_path="seqvec/uniref50_v2"):
    embs = seqvec_embedder.list_embeddings(sequences, seqvec_path)
    embs_residue = seqvec_embedder.list_embeddings_residue(sequences, seqvec_path)
    return embs, embs_residue

def embed(pos_seqs, neg_seqs, models_dir, seqvec_path="seqvec/uniref50_v2"):
    pos_embeddings = seqvec_embedder.list_embeddings(pos_seqs, seqvec_path)
    neg_embeddings = seqvec_embedder.list_embeddings(neg_seqs, seqvec_path)
    pos_embeddings_residue = seqvec_embedder.list_embeddings_residue(pos_seqs, seqvec_path)
    neg_embeddings_residue = seqvec_embedder.list_embeddings_residue(neg_seqs, seqvec_path)
    
    X = np.concatenate((np.asarray(pos_embeddings), np.asarray(neg_embeddings)))
    y = np.concatenate((np.ones(len(pos_seqs)), np.zeros(len(neg_seqs))))

    X_residue = np.concatenate((np.asarray(pos_embeddings_residue), np.asarray(neg_embeddings_residue)))
    y_residue = np.concatenate((np.ones(len(pos_seqs)), np.zeros(len(neg_seqs))))

    print("Input per protein shape:", X.shape)
    print("Input per residue shape:", X_residue.shape)
    
    # shuffle
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    X_residue, y_residue = X_residue[indices], y_residue[indices]

    save_base_models(X, y, models_dir)
    model = models.lstm_model(EMB_LEN, TIMESTEPS, ENCODING_DIM)
    history = model.fit(X_residue, y_residue, epochs=10, validation_split=0.25, batch_size=10)
    model.save("{}/{}.h5".format(models_dir, SEQ_MODEL_NAME))
    print("Saved SeqVec sequential embedding model in", models_dir)

def predict(embs, embs_residue, models_dir):
    # per residue predictions
    lstm_modelname = "{}{}.h5".format(models_dir, SEQ_MODEL_NAME)
    y_pred_res_score, y_pred_res = models.predict_lstm(lstm_modelname, embs_residue)
    # base predictions
    y_pred_base = models.predict_base(models_dir, embs)
    # stacking predictions
    y_pred_stacking = models.predict_stacking(models_dir, embs)

    predictions = np.column_stack((y_pred_res_score, y_pred_base, y_pred_stacking))
    final_preds = np.average(predictions, axis=1)
    return final_preds

def one_hot_enconding(sequences):
    def findUniqueAminoAcids(sequences):
        aa = set()
        for seq in sequences:
            for char in seq:
                aa.add(char)
        aminoAcids = [i for i in aa]
        aminoAcids = sorted(aminoAcids)
        indexAAs = {}
        for i, ch in enumerate(aminoAcids):
            indexAAs[ch] = i
        return aminoAcids, indexAAs
    aminoAcids, indexAAs = findUniqueAminoAcids(sequences)

    n = len(sequences)
    d = len(aminoAcids)
    X = np.zeros((n, TIMESTEPS, d))

    for i in range(n):
        sequence = sequences[i]
        for index in range(len(sequence)):
            ch = sequence[index]
            X[i, index, indexAAs[ch]] = 1
    return X