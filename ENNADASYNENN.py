from imblearn.under_sampling import EditedNearestNeighbours
from Adasyn import ADASYN
import numpy as np
from NearestNeighboursExactDenoise import NearestNeighboursExactDenoise

class EnnAdasynEnn:
    def __init__(self, sampling_strategy='minority', n_neighbors=5, random_state = 10):
        self.resample_strategy = sampling_strategy
        self.k_nearest_neighbours = n_neighbors
        self.random_state = random_state

    def isLengthEqual(self):
        separated_x = self.separate_x()
        length = len(separated_x)

        temp = len(separated_x[0])

        for i in range(length):
            if abs(temp - len(separated_x[i])) > 1:
                return False

        return True

    def getLength(self):
        separated_x = self.separate_x()
        length = len(separated_x)
        for i in range(length):
            print(i, " length:", len(separated_x[i]))

    def get_ration(self):
        separated_x = self.separate_x()
        length = len(separated_x)

        majority_class_length = len(separated_x[0])
        min_class_length = len(separated_x[0])

        for i in range(length):
            if len(separated_x[i]) > majority_class_length:
                majority_class_length = len(separated_x[i])
            if len(separated_x[i]) < min_class_length:
                min_class_length = len(separated_x[i])

        return majority_class_length/min_class_length

    def fit_resample(self, X, y):
        count = 0
        self.x = X
        self.y = y
        length_of_majority = self.get_majority_sample_count()

        self.enn_resample()

        while (self.isLengthEqual() == False and count < 10):
            self.ADASYN_resample()
            self.enn_resample()
            count += 1

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

    def get_majority_sample_count(self):
        separated_x = self.separate_x()

        length = 0

        if len(separated_x[0]) > len(separated_x[1]):
            length = len(separated_x[0])
        else:
            length = len(separated_x[1])

        return length

    def ADASYN_resample(self):
        # adasyn_ratio = self.get_ration()
        adasyn = ADASYN(ratio=0, n_neighbors=self.k_nearest_neighbours)
        self.x, self.y = adasyn.fit_resample(self.x, self.y)

    def enn_resample(self):
        if self.resample_strategy == [1]:
            self.enn_majority_resample()

    def enn_majority_resample(self):
        enn = EditedNearestNeighbours(sampling_strategy=[1], n_neighbors=self.k_nearest_neighbours)
        self.x, self.y = enn.fit_resample(self.x, self.y)

    def get_params(self):
        return {'n_neighbors'}