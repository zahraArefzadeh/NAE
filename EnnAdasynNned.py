from imblearn.under_sampling import EditedNearestNeighbours
from Adasyn import ADASYN
import numpy as np
from NearestNeighboursExactDenoise import NearestNeighboursExactDenoise


class EnnAdasynNned:
    def __init__(self, sampling_strategy='minority', n_neighbors=5, random_state = 10):
        self.resample_strategy = sampling_strategy
        self.k_nearest_neighbours = n_neighbors
        self.random_state = random_state

    def isLengthEqual(self):
        separated_x = self.separate_x()
        length = len(separated_x)

        temp = len(separated_x[1])

        for i in range(length):
            print(i, " length:", len(separated_x[i]))

        for i in range(length):
            if abs(temp - len(separated_x[i])) > 3:
                return False

        return True

    def fit_resample(self, X, y):
        count = 0
        self.x = X
        self.y = y

        self.enn_resample()
        try:
            while (self.isLengthEqual() == False and count < 10):
                self.ADASYN_resample()
                self.modified_enn_resample()
                count += 1

            print("number of iterations: ", count)
        except:
            print("test")

        return self.x, self.y

    def separate_x(self):
        """Separates the x samples into lists based on their corresponding class labels.

        Returns:
            dict: A dictionary where keys are the unique class labels and values are lists of x samples for each class.
        """
        unique_classes = np.unique(self.y)  # Get unique class labels

        class_data = {}
        for class_label in unique_classes:
            mask = self.y == class_label
            class_data[class_label] = self.x[mask]

        return class_data

    def ADASYN_resample(self):
        adasyn = ADASYN(ratio=0, n_neighbors=self.k_nearest_neighbours)
        self.x, self.y = adasyn.fit_resample(self.x, self.y)

    def enn_resample(self):
        if self.resample_strategy == [1]:
            self.enn_majority_resample()

    def enn_majority_resample(self):
        enn = EditedNearestNeighbours(sampling_strategy=[1], n_neighbors=self.k_nearest_neighbours)
        self.x, self.y = enn.fit_resample(self.x, self.y)

    def modified_enn_resample(self):
        enn = NearestNeighboursExactDenoise('minority', 5)
        self.x, self.y = enn.fit_resample(self.x, self.y)