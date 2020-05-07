"""
Date: 2020-01-15
Author: figalit (github.com/figalit)

Contains definitions for models used in tAMPer.
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from keras.models import Sequential, Model
from keras.layers.core import Masking
from keras.layers import Bidirectional, LSTM, RepeatVector, TimeDistributed, Dense, Embedding, Dropout
from keras import regularizers
from keras.optimizers import Adam

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
    stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    return clf1, clf2, clf3, stacking

def predict_base(models_dir, embeddings):
    clf1, clf2, clf3, stacking = base_models()
    preds = []    
    for clf in [clf1,clf2,clf3, stacking]:
        name = type(clf).__name__
        clf = pickle.load(open("{}{}.sav".format(models_dir, name), 'rb'))
        p = clf.predict_proba(data)[:,1]
        preds.append(p)
    return np.stack(tuple(preds), axis=0).T


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
    model = keras.models.load_model(modelname)
    y_pred_scores = model.predict(embeddings)
    y_pred = []
    for i in y_pred_scores:
        if i>=0.5: y_pred.append(1)
        else: y_pred.append(0)
    return y_pred_scores, y_pred

