import numpy as np
from sklearn.neighbors import NearestNeighbors


class ADASYN:
    def __init__(self, ratio=0.5, n_neighbors=5, random_state=42):
        self.ratio = ratio
        self.k = n_neighbors
        self.random_state = random_state

    def get_params(self):
        return {'n_neighbors'}

    def fit_resample(self, X, y):
        self.X_resampled = X.copy()  # Preserve original data
        self.y_resampled = y.copy()

        minority_class = 1  # Find minority class
        minority_indices = np.where(y == 1)[0]

        length_of_majority_class = y.shape[0] - len(minority_indices)

        G = int(length_of_majority_class - len(minority_indices))

        nn = NearestNeighbors(n_neighbors=self.k + 1)
        
        nn.fit(X)

        _, knn_indices = nn.kneighbors(X[minority_indices])

        ri = 1-(np.sum(y[knn_indices[:, 1:]] , axis=1) / self.k)#!!!   3/5--->2/5   1-3/5=2/5
        if(np.sum(ri)>0):
            ri=ri/np.sum(ri)
        gi = np.round(ri * G)
        
        #add
        nn = NearestNeighbors(n_neighbors=self.k + 1)
        nn.fit(X[y==1])
        _, knn_indices = nn.kneighbors(X[minority_indices])
        #---
        for i, g in enumerate(gi):
            if g > 0:
                # print(gi)
                #!!!!اگر یه نمونه صفر شد چی
                minority_sample = X[minority_indices[i]]
                _, knn_indices = nn.kneighbors([minority_sample])
                # print(knn_indices)
                #nearest neighbors of i
                
                for _ in range(int(g)):
                    nn_index=np.random.choice(np.arange(1,self.k+1))
                    minor_indx=knn_indices[0][nn_index]
                    neighbour=np.array(X[minority_indices][minor_indx])
                    # print(nn_index)
                    # print('====')
                    synthetic_sample = minority_sample + np.random.random() * (neighbour - minority_sample)
                    self.X_resampled = np.vstack([self.X_resampled, synthetic_sample])
                    self.y_resampled = np.append(self.y_resampled, minority_class)

        return self.X_resampled, self.y_resampled