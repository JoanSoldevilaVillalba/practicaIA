__authors__ = ''
__group__ = 'TO_BE_FILLED'

import numpy as np
import utils
from scipy.spatial.distance import cdist

class KMeans:

    def __init__(self, X, K=1, options=None):
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)

    def _init_X(self, X):
        X_array = np.array(X, dtype=np.float64)
        if len(X_array.shape) > 2:
            self.X = X_array.reshape(-1, X_array.shape[-1])
        else:
            self.X = X_array

    def _init_options(self, options=None):
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'
        self.options = options

    def _init_centroids(self):
        self.old_centroids = np.zeros((self.K, self.X.shape[1]), dtype=np.float64)
        if self.options['km_init'].lower() == 'first':
            _, idx = np.unique(self.X, axis=0, return_index=True)
            idx = np.sort(idx)
            self.centroids = self.X[idx[:self.K]].copy()
        elif self.options['km_init'].lower() == 'random':
            idx = np.random.choice(self.X.shape[0], self.K, replace=False)
            self.centroids = self.X[idx].copy()
        else:
            self.centroids = np.random.rand(self.K, self.X.shape[1]).astype(np.float64)

    def get_labels(self):
        dist = distance(self.X, self.centroids)
        self.labels = np.argmin(dist, axis=1)

    def get_centroids(self):
        self.old_centroids = self.centroids.copy()
        for k in range(self.K):
            punts_k = self.X[self.labels == k]
            if len(punts_k) > 0:
                self.centroids[k] = np.mean(punts_k, axis=0)

    def converges(self):
        return np.allclose(self.centroids, self.old_centroids, atol=self.options['tolerance'])

    def fit(self):
        self._init_centroids()
        self.num_iter = 0
        while self.num_iter < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1
            if self.converges():
                break

    def withinClassDistance(self):
        WCD = 0.0
        for k in range(self.K):
            punts_k = self.X[self.labels == k]
            if len(punts_k) > 0:
                dist_quadrat = np.sum((punts_k - self.centroids[k])**2)
                WCD += dist_quadrat
        self.WCD = WCD / self.X.shape[0]
        return self.WCD

    def find_bestK(self, max_K):
        wcd_hist = []
        for k in range(1, max_K + 1):
            self.K = k
            self.fit()
            wcd_actual = self.withinClassDistance()
            wcd_hist.append(wcd_actual)
            if len(wcd_hist) > 1:
                dec = 100 * (wcd_hist[-1] / wcd_hist[-2])
                if (100 - dec) < 20:
                    self.K = k - 1
                    return
        self.K = max_K

def distance(X, C):
    return cdist(X, C, metric='euclidean')

def get_colors(centroids):
    probs = utils.get_color_prob(centroids)
    color_indexs = np.argmax(probs, axis=1)
    return [utils.colors[i] for i in color_indexs]