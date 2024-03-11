import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import seaborn as sns
from sklearn.cluster import KMeans
from NnedAdasynEnn import NnedAdasynEnn
from ReverseAdasyn import REVERSEADASYN
from ADASYNENN import ADASYNENN
from ENNADASYNENN import EnnAdasynEnn
from AdasynNned import AdasynNned
from EnnAdasynNned import EnnAdasynNned
from imblearn.over_sampling import SMOTE,ADASYN,KMeansSMOTE,RandomOverSampler
from Adasyn import ADASYN as ourADASYN

n_samples = 1100
centers = [[-0.15, 0], [0.12, 0]]
cluster_std = [0.08, 0.065]
X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=123)
#------------------------------------------------------------------------------
overSamplingMethods=[SMOTE,ADASYN,ourADASYN,NnedAdasynEnn,REVERSEADASYN,ADASYNENN,EnnAdasynEnn,AdasynNned,EnnAdasynNned]

#------------------------------------------------------------------------------
# Delete half of the samples with label 1
indices_to_delete = np.array(np.where(y == 1)[0])[0:350]
X = np.delete(X, indices_to_delete, axis=0)
y = np.delete(y, indices_to_delete, axis=0)
#------------------------------------------------------------------------------
n_small_disjunct=7
samples, labels = make_blobs(n_samples=n_small_disjunct, centers=[[-0.22, -0.05]], cluster_std=0.012, random_state=123)
labels=np.ones(n_small_disjunct)
#------------------------------------------------------------------------------
n_small_cluster=50
X = np.concatenate((X, samples), axis=0)
y = np.concatenate((y, labels), axis=0)
samples, labels = make_blobs(n_samples=n_small_cluster, centers=[[-0.35, 0.14]], cluster_std=0.05, random_state=123)
labels=np.ones(n_small_cluster)
X = np.concatenate((X, samples), axis=0)
y = np.concatenate((y, labels), axis=0)
#------------------------------------------------------------------------------
samples, labels = make_blobs(n_samples=8, centers=[[-0.16, -.06]], cluster_std=0.012, random_state=123)
labels=np.ones(8)
X = np.concatenate((X, samples), axis=0)
y = np.concatenate((y, labels), axis=0)

#------------------------------------------------------------------------------
X=np.array(X.astype(float))
y=np.array(y.astype(int))
#------------------------------------------------------------------------------
# Perform  oversampling
for overSamplingMethod in overSamplingMethods:
    overSampler = overSamplingMethod()
    
    # X_resampled, y_resampled =X,y
    X_resampled, y_resampled = overSampler.fit_resample(X, y)
    
    X_resampled=np.array(X_resampled.astype(float))
    #------------------------------------------------------------------------------
    X_new_indices = []
    for x in X:
        index = np.where(X_resampled == x)[0]
        if index.size > 0:
            X_new_indices.append(index[0])  # Append the first occurrence if the value is found
    
    X_new=X_resampled[X_new_indices]
    y_new=y_resampled[X_new_indices]
    #------------------------------------------------------------------------------
    indices_synthetic =np.unique( np.where(~np.isin(X_resampled, X_new))[0])
    
    X_synthetic=X_resampled[indices_synthetic]
    y_synthetic=y_resampled[indices_synthetic]
    #------------------------------------------------------------------------------
    colors = {0: 'red', 1: 'blue', 2: 'green'}  # Green for synthetic samples
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 6),dpi=300)
    
    # Plot minority-class samples in red
    sns.scatterplot(x=X_new[y_new == 0][:, 0], y=X_new[y_new == 0][:, 1], color=colors[0], alpha=0.6, label="Majority Class")
    
    # Plot majority-class samples in blue
    sns.scatterplot(x=X_new[y_new != 0][:, 0], y=X_new[y_new != 0][:, 1], color=colors[1], alpha=0.6, label="Minority Class")
    
    # Plot synthetic minority-class samples in green
    sns.scatterplot(x=X_synthetic[:, 0], y=X_synthetic[:, 1], color=colors[2], alpha=0.8, label="Synthetic Minority Samples").set_title(overSamplingMethod)
    
    # Plot noisy minority-class samples with a distinct marker
    # sns.scatterplot(x=X_synthetic[:, 0], y=X_synthetic[:, 1], color=colors[1], marker='X', s=200, label="Noisy Minority Samples")
    
    plt.legend(loc="best")
    
    plt.tight_layout()
    plt.show()
