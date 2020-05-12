"""
Date: 2020-01-15
Author: figalit (github.com/figalit)

Contains definitions for models used in tAMPer.
"""
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from keras.models import Sequential, Model
from keras.layers.core import Masking
from keras.layers import Bidirectional, LSTM, RepeatVector, TimeDistributed, Dense, Embedding, Dropout
from keras import regularizers
from keras.optimizers import Adam
from keras.models import load_model
import pandas as pd
import numpy as np
import pickle
import os
NFOLD = 3

def base_models():
    """Returns the current base models used in the classification of SeqVec vectors"""
    clf1 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
    clf2 = LogisticRegression(random_state=0)
    clf3 = RandomForestClassifier(max_depth=15, random_state=0, n_estimators=300)
    
    estimators = [
        ("gboost", clf1),
        ("logreg", clf2),
        ("rf", clf3),
    ]
    return clf1, clf2, clf3

def predict_base(models_dir, embeddings):
    clf1, clf2, clf3 = base_models()
    preds = []    
    for clf in [clf1,clf2,clf3]:
        name = type(clf).__name__
        clf = pickle.load(open("{}{}.sav".format(models_dir, name), 'rb'))
        p = clf.predict_proba(embeddings)[:,1]
        preds.append(p)
    return np.stack(tuple(preds), axis=0).T

def stacking(model, X, y):
    X, y = pd.DataFrame(X), pd.DataFrame(y)
    folds = StratifiedKFold(n_splits=NFOLD, random_state=1)

    train_pred = np.empty((0,1),float)
    train_y = np.empty((0,1),float)
    for train_indices, val_indices in folds.split(X,y):
        x_train,x_val=X.iloc[train_indices],X.iloc[val_indices]
        y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]
        y_train = y_train.values.ravel()
        model.fit(X=x_train,y=y_train)
        train_pred=np.append(train_pred,model.predict_proba(x_val)[:,1])
        train_y=np.append(train_y,y_val)
    return train_pred, train_y

def save_stacking(model, X, y, models_dir):
    X, y = pd.DataFrame(X), pd.DataFrame(y) 
    name = type(model).__name__
    folds = StratifiedKFold(n_splits=NFOLD, random_state=1)
    train_pred = np.empty((0,1),float)
    
    for train_indices, val_indices in folds.split(X,y):
        x_train,x_val=X.iloc[train_indices],X.iloc[val_indices]
        y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]
        y_train = y_train.values.ravel()
        model.fit(X=x_train,y=y_train)
        train_pred=np.append(train_pred,model.predict_proba(x_val)[:,1])
    y = y.values.ravel()
    model.fit(X, y)

    stacking_dir = "{}/stacking/".format(models_dir)
    if not os.path.exists(stacking_dir): os.makedirs(stacking_dir)
    pickle.dump(model, open("{}stacking_{}.sav".format(stacking_dir, name), 'wb'))

def stacking_fit(X, y, models_dir):
    clf1, clf2, clf3 = base_models()
    save_stacking(clf1, X, y, models_dir)
    save_stacking(clf2, X, y, models_dir)
    save_stacking(clf3, X, y, models_dir)

    trainDfs, y_train_new = [], []
    for clf in [clf1,clf2,clf3]:
        train, y_train = stacking(clf, X, y)
        trainDfs.append(pd.DataFrame(train))
        y_train_new = y_train
    df_train = pd.concat(trainDfs, axis=1)

    model = LogisticRegression(random_state=42)
    model.fit(df_train, y_train_new.ravel())
    return model

def predict_stacking(models_dir, embeddings):
    # Load stacking models and prepare data.
    preds = []
    clf1, clf2, clf3 = base_models()
    for c in [clf1,clf2,clf3]:
        name = type(c).__name__
        clf = pickle.load(open("{}/stacking/stacking_{}.sav".format(models_dir, name), 'rb'))
        p = clf.predict_proba(embeddings)[:,1]
        preds.append([p])
    top = pickle.load(open("{}/stacking/StackingClassifier.sav".format(models_dir), 'rb')) 
    preds = np.concatenate(preds, axis=0).T
    return top.predict_proba(preds)[:,1]

def lstm_model(features, timesteps, encoding_dim):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(timesteps,features)))
    model.add(Bidirectional(LSTM(encoding_dim)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(encoding_dim, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

def predict_lstm(modelname, embeddings):
    model = load_model(modelname)
    y_pred_scores = model.predict(embeddings)
    y_pred = []
    for i in y_pred_scores:
        if i>=0.5: y_pred.append(1)
        else: y_pred.append(0)
    return y_pred_scores, y_pred

