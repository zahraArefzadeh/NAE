import numpy as np
from sklearn.neighbors import NearestNeighbors

class REVERSEADASYN:
    def __init__(self, k=5):
        self.k = k

    def fit_resample(self, X, y):
        self.X_resampled = X.copy()  # Preserve original data
        self.y_resampled = y.copy()

        minority_class = np.argmin(np.bincount(y))  # Find minority class
        minority_indices = np.where(y == minority_class)[0]

        length_of_majority_class = y.shape[0] - np.sum(y == minority_class)

        G = int(length_of_majority_class - np.sum(y == minority_class))

        nn = NearestNeighbors(n_neighbors=self.k + 1)
        nn.fit(X)

        _, knn_indices = nn.kneighbors(X[minority_indices])

        ri = 1 - np.sum(y[knn_indices[:, 1:]] != minority_class, axis=1) / self.k
        # ri = np.sum(y[knn_indices[:, 1:]] != minority_class, axis=1) / self.k
        gi = np.nan_to_num(np.floor(ri * G / (np.sum(ri) + 0.000001)))

        print(G)
        print(np.sum(gi))

        for i, g in enumerate(gi):
            if g > 0:
                minority_sample = X[minority_indices[i]]
                knn = X[knn_indices[i, 1:]]
                for _ in range(int(g)):
                    nn_index = np.random.choice(self.k, 1)[0]
                    synthetic_sample = minority_sample + np.random.random() * (knn[nn_index] - minority_sample)
                    self.X_resampled = np.vstack([self.X_resampled, synthetic_sample])
                    self.y_resampled = np.append(self.y_resampled, minority_class)

        return self.X_resampled, self.y_resampled
