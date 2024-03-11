from AdasynEnn import ADASYNENN
from EnnAdasynEnn import EnnAdasynEnn
from NnedAdasynEnn import NnedAdasynEnn
from AdasynNned import AdasynNned
from NnedAdasynNned import NnedAdasynNned
from EnnAdasynNned import EnnAdasynNned

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.datasets import load_iris
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import RandomForestClassifier
from NearestNeighboursExactDenoise import NearestNeighboursExactDenoise

randomState = 46
nNeighbors = 5
samplingStrategy = 'minority'

data = pd.read_csv("C:/Users/erfan/Desktop/BinaryDatasets/segment0.dat", header=None)
data.iloc[:, -1] = pd.factorize(data.iloc[:, -1])[0]

# Loading targets
X = np.array(data.iloc[:, :-1].values)
# Loading Data
y = np.array(data.iloc[:, -1].values)
y = y.astype(int)

x_train, x_test, y_train, y_test = train_test_split(X,y , test_size=0.2, random_state=randomState, shuffle=True)

# ADASYN RESAMPLE
adasyn = ADASYN(sampling_strategy=samplingStrategy, random_state=randomState, n_neighbors=nNeighbors)
adasyn_x_train, adasyn_y_train = adasyn.fit_resample(x_train, y_train)

adasynEnn = ADASYNENN(sampling_strategy=samplingStrategy, n_neighbors=nNeighbors, random_state=randomState)
adasynEnn_x_train, adasynEnn_y_train = adasynEnn.fit_resample(x_train, y_train)

ennAdasynEnn = EnnAdasynEnn(sampling_strategy=samplingStrategy, n_neighbors=nNeighbors, random_state=randomState)
ennAdasynEnn_x_train, ennAdasynEnn_y_train = ennAdasynEnn.fit_resample(x_train, y_train)

nnedAdasynEnn = NnedAdasynEnn(sampling_strategy=samplingStrategy, n_neighbors=nNeighbors, random_state=randomState)
nnedAdasynEnn_x_train, nnedAdasynEnn_y_train = nnedAdasynEnn.fit_resample(x_train, y_train)

adasynNned = AdasynNned(sampling_strategy=samplingStrategy, n_neighbors=nNeighbors, random_state=randomState)
adasynNned_x_train, adasynNned_y_train = adasynNned.fit_resample(x_train, y_train)

nnedAdasynNned = NnedAdasynNned(sampling_strategy=samplingStrategy, n_neighbors=nNeighbors, random_state=randomState)
nnedAdasynNned_x_train, nnedAdasynNned_y_train = nnedAdasynNned.fit_resample(x_train, y_train)

randomForestClassifier = RandomForestClassifier(n_estimators=100, random_state=randomState)

print("-----------------------------------------------------------------")

randomForestClassifier.fit(adasyn_x_train, adasyn_y_train)
adasyn_y_predict = randomForestClassifier.predict(x_test)
print("F1-score ADASYN:", f1_score(y_test, adasyn_y_predict))
print("ROC ADASYN:", roc_auc_score(y_test, adasyn_y_predict))

print("-----------------------------------------------------------------")

randomForestClassifier.fit(adasynEnn_x_train, adasynEnn_y_train)
adasynEnn_y_predict = randomForestClassifier.predict(x_test)
print("F1-score ADASYN-ENN:", f1_score(y_test, adasynEnn_y_predict))
print("ROC ADASYN-ENN:", roc_auc_score(y_test, adasynEnn_y_predict))

print("-----------------------------------------------------------------")

randomForestClassifier.fit(ennAdasynEnn_x_train, ennAdasynEnn_y_train)
ennAdasynEnn_y_predict = randomForestClassifier.predict(x_test)
print("F1-score ENN-ADASYN-ENN:", f1_score(y_test, ennAdasynEnn_y_predict))
print("ROC ENN-ADASYN-ENN:", roc_auc_score(y_test, ennAdasynEnn_y_predict))

print("-----------------------------------------------------------------")

randomForestClassifier.fit(nnedAdasynEnn_x_train, nnedAdasynEnn_y_train)
NnedAdasynEnn_y_predict = randomForestClassifier.predict(x_test)
print("F1-score NNED-ADASYN-ENN:", f1_score(y_test, NnedAdasynEnn_y_predict))
print("ROC NNED-ADASYN-ENN:", roc_auc_score(y_test, NnedAdasynEnn_y_predict))

print("-----------------------------------------------------------------")

randomForestClassifier.fit(adasynNned_x_train, adasynNned_y_train)
adasynNned_y_predict = randomForestClassifier.predict(x_test)
print("F1-score ADASYN-NNED:", f1_score(y_test, adasynNned_y_predict))
print("ROC ADASYN-NNED:", roc_auc_score(y_test, adasynNned_y_predict))

print("-----------------------------------------------------------------")

randomForestClassifier.fit(nnedAdasynNned_x_train, nnedAdasynNned_y_train)
nnedAdasynNned_y_predict = randomForestClassifier.predict(x_test)
print("F1-score NNED-ADASYN-NNED:", f1_score(y_test, nnedAdasynNned_y_predict))
print("ROC NNED-ADASYN-NNED:", roc_auc_score(y_test, nnedAdasynNned_y_predict))
