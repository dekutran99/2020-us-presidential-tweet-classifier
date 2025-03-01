import numpy as np
from utils import euclidean_dist_squared

class Kmeans:

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)

        means = np.zeros((self.k, D))
        for kk in range(self.k):
            i = np.random.randint(N)
            means[kk] = X[i]

        while True:
            y_old = y

            # Compute euclidean distance to each mean
            dist2 = euclidean_dist_squared(X, means)
            dist2[np.isnan(dist2)] = np.inf
            y = np.argmin(dist2, axis=1)

            # Update means
            for kk in range(self.k):
                if np.any(y==kk): # don't update the mean if no examples are assigned to it (one of several possible approaches)
                    means[kk] = X[y==kk].mean(axis=0)

            changes = np.sum(y != y_old)
            # print('Running K-means, changes in cluster assignment = {}'.format(changes))

            # Stop if no point changed cluster
            if changes == 0:
                break

        self.means = means

    def predict(self, X):
        means = self.means
        dist2 = euclidean_dist_squared(X, means)
        dist2[np.isnan(dist2)] = np.inf
        return np.argmin(dist2, axis=1)

    # Compute error of prediction using sum of squared distance between examples
    def error(self, X):
        # get number of samples and features in X
        N, D = X.shape                

        # k cluster centroids
        means = self.means

        # predict the cluster for each sample
        w = self.predict(X)
        w = means[w]

        # compute error using sum[i = 1, n]sum[j = 1, d]((x_ij - w_ij)**2)
        dist2 = (X - w)**2      # euclidean_dist_squared(X, w) does not do the same thing?
        error = np.sum(dist2)

        return error