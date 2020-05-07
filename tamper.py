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

def embed(pos_seqs, neg_seqs, seqvec_path="seqvec/uniref50_v2", models_dir):
    pos_embeddings = seqvec_embedder.list_embeddings(pos_seqs)
    neg_embeddings = seqvec_embedder.list_embeddings(neg_seqs)    
    pos_embeddings_residue = seqvec_embedder.list_embeddings_residue(pos_seqs)
    neg_embeddings_residue = seqvec_embedder.list_embeddings_residue(neg_seqs)
    
    X = np.concatenate((np.asarray(pos_embeddings), np.asarray(neg_embeddings)))
    y = np.concatenate((np.ones(len(pos_seqs)), np.zeros(len(neg_seqs))))

    # TODO: May need to adjust this? 
    X_residue = np.concatenate((np.asarray(pos_embeddings_residue), np.asarray(neg_embeddings_residue)))

    # TODO: Store embeddings?

    models.save_base_models(X, y, models_dir)

    model = models.lstm_model(EMB_LEN, TIMESTEPS, ENCODING_DIM)
    history = model.fit(X_residue, y, epochs=10, validation_split=0.25, batch_size=10)
    model.save("{}{}.h5".format(models_dir, "lstm_model"))

    print("Models saved in", models_dir)

def predict(embs, embs_residue, models_dir):
    # per residue predictions
    lstm_modelname = "{}{}.h5".format(models_dir, "lstm_model")
    y_pred_res_score, y_pred_res = models.predict_lstm(lstm_modelname, embs_residue)
    # base predictions
    y_pred_base = models.predict_base(models_dir, embs)
    
    predictions = np.column_stack((y_pred_res_score, y_pred_base))
    final_preds = np.average(predictions, axis=1)
    return final_preds