__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'
import numpy as np
from scipy.spatial.distance import cdist
from scipy import stats


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)

    def _init_train(self, train_data):
        self.train_data = train_data.reshape(
            train_data.shape[0], -1).astype(np.float64)

    def get_k_neighbours(self, test_data, k):
        if len(test_data.shape) > 2:
            test_data_flat = test_data.reshape(
                test_data.shape[0], -1).astype(np.float64)
        else:
            test_data_flat = test_data.astype(np.float64)

        distancies = cdist(test_data_flat, self.train_data, metric='euclidean')
        index_ordenats = np.argsort(distancies, axis=1)[:, :k]
        self.neighbors = self.labels[index_ordenats]
        self.neighbour_index = index_ordenats
        distancies_ordenades = np.sort(distancies, axis=1)[:, :k]
        self.distancies = distancies_ordenades
        # aqui haurem de definir un nou atribut de la class KNN
        # cal tenir en compte diverses coses, primerament index_ordenats esta en un ordre completament diferent que els veins

    def get_class(self):
        P_test = self.neighbors.shape[0]
        predictions = np.empty(P_test, dtype=self.labels.dtype)
        for i in range(P_test):
            veins_actuals = self.neighbors[i]
            valors, comptes = np.unique(veins_actuals, return_counts=True)
            max_vots = np.max(comptes)
            guanyadors = valors[comptes == max_vots]
            if len(guanyadors) > 1:
                for v in veins_actuals:
                    if v in guanyadors:
                        predictions[i] = v
                        break
            else:
                predictions[i] = guanyadors[0]
        return predictions

    def predict(self, test_data, k):
        self.get_k_neighbours(test_data, k)
        return self.get_class()
