import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from imblearn.over_sampling import SMOTE,ADASYN,KMeansSMOTE,RandomOverSampler
import smote_variants as sv
from NnedAdasynEnn import NnedAdasynEnn
from ReverseAdasyn import REVERSEADASYN
from ADASYNENN import ADASYNENN
from ENNADASYNENN import EnnAdasynEnn
from AdasynNned import AdasynNned
from EnnAdasynNned import EnnAdasynNned
from Adasyn import ADASYN as ourADASYN
# Create the Three-Cluster Dataset with Noisy Minority-Class Samples
n_samples = 400
centers = [[1,1], [0.5, 0], [2, 2]]
cluster_std = [0.3, 0.2, 0.3]
X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=32)

y[y != 0] = 1
#------------------------------------------------------------------------------
overSamplingMethods=[SMOTE,ADASYN,ourADASYN,NnedAdasynEnn,REVERSEADASYN,ADASYNENN,EnnAdasynEnn,AdasynNned,EnnAdasynNned]
#------------------------------------------------------------------------------
indx_y_1=np.where(y==1)[0][:4]
X=np.delete(X,indx_y_1,axis=0)
y=np.delete(y,indx_y_1,axis=0)
y = 1 - y

# Add noisy minority-class samples surrounded by majority-class samples
noisy_samples = np.array([[0, 0.5], [1.7, 2], [2.5, 2.5],])
noisy_labels = np.array([1, 1, 1])  # Noisy samples labeled as minority class
X = np.concatenate((X, noisy_samples), axis=0)
y = np.concatenate((y, noisy_labels), axis=0)
#------------------------------------------------------------------------------
# Perform  oversampling
for overSamplingMethod in overSamplingMethods:
    overSampler = overSamplingMethod()
    X_resampled, y_resampled =X,y
    X_resampled, y_resampled = overSampler.fit_resample(X, y)
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
    
    # Define colors for minority and majority classes
    colors = {0: 'red', 1: 'blue', 2: 'green'}  # Green for synthetic samples
    
    # Plot the Three-Cluster Dataset with Noisy Samples using Seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 6),dpi=300)
    
    
    
    # Plot minority-class samples in red
    sns.scatterplot(x=X_new[y_new == 0][:, 0], y=X_new[y_new == 0][:, 1], color=colors[0], alpha=0.8, label="Majority Class")
    
    # Plot majority-class samples in blue
    sns.scatterplot(x=X_new[y_new != 0][:, 0], y=X_new[y_new != 0][:, 1], color=colors[1], alpha=0.9, label="Minority Class")
    
    # Plot synthetic minority-class samples in green
    sns.scatterplot(x=X_synthetic[:, 0], y=X_synthetic[:, 1], color=colors[2], alpha=0.8, label="Synthetic Minority Samples").set_title(overSamplingMethod)
    
    # Plot noisy minority-class samples with a distinct marker
    sns.scatterplot(x=noisy_samples[:, 0], y=noisy_samples[:, 1], color=colors[1], marker='X', s=200, label="Noisy Minority Samples")
    
    plt.legend(loc="best")
    
    plt.tight_layout()
    plt.show()
    
    
    
