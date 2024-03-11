import pandas as pd
import numpy as np
from  sklearn.datasets import load_breast_cancer
from sklearn.neighbors import NearestNeighbors



class NearestNeighboursExactDenoise :

    def __init__ (self,sampling_strategy,n_neighbors):

       self.sampling_strategy = sampling_strategy
       self.n_neighbors = n_neighbors


    def class_divider(self):
      y=self.y

      positive_samples = y[y == 1].copy()
      negative_samples = y[y == 0].copy()

      if len(positive_samples) > len(negative_samples) :
         majority_class = positive_samples
         minority_class = negative_samples
      else :
         majority_class = negative_samples
         minority_class = positive_samples


      return minority_class , majority_class

    def set_enn_class(self):
        minority_class , majority_class = self.class_divider()
        X = self.X
        if self.sampling_strategy == "all" :
          return X
        elif self.sampling_strategy == "minority":
            return X[self.y==1]
        elif self.sampling_strategy == "majority":
            return X[self.y == 0]


    def fit_resample(self, X_train, y_train):
        self.X = X_train
        self.y = y_train

        enn_class=self.set_enn_class()

        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nbrs.fit(self.X,self.y)
        indices = nbrs.kneighbors(X=enn_class, return_distance=False)

        y = np.array(self.y)
        y_neighbors = y[indices]

        y_neighbors_without_self = y_neighbors[:,1:]

        y_neighbors_sum = np.sum(y_neighbors_without_self,axis=1)

        samples_to_be_removed = indices[:, 0][y_neighbors_sum==0]

        X = np.delete(self.X, samples_to_be_removed, axis=0)
        y = np.delete(self.y, samples_to_be_removed, axis=0)

        return X, y 