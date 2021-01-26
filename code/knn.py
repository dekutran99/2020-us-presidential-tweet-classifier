"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils


class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the training data
        self.y = y

    def predict(self, Xtest):
        X = self.X
        y = self.y
        n = X.shape[0]
        t = Xtest.shape[0]
        k = min(self.k, n)

        # Compute cosine_distance distances between X and Xtest
        dist2 = cosine_distance(X, Xtest)

        # yhat is a vector of size t with integer elements
        yhat = np.ones(t, dtype=np.uint8)
        for i in range(t):
            # sort the distances to other points
            inds = np.argsort(dist2[:,i])

            # compute mode of k closest training pts
            yhat[i] = stats.mode(y[inds[:k]])[0][0]

        return yhat

def cosine_distance(X1,X2):
    '''
    if there is any zero row in X1 or X2, set the distance between any zero row in X1 to all the rows X2 to zero or vice versa.
    '''
    cos_dist = np.ones([X1.shape[0], X2.shape[0]])
    for i in range(X1.shape[0]):
        cosine_similarity =np.ones(X2.shape[0])
        for j in range(X2.shape[0]):
            # v1 is vector from X1
            v1 = X1[i]
            # v2 is vector from X2
            v2 = X2[j]
            # v1 norm
            v1_norm = np.linalg.norm(v1, ord=2)
            # v2 norm
            v2_norm = np.linalg.norm(v2, ord=2)
            

            # compute cosine similarity between v1 and v2
            if v1_norm == 0 or v2_norm == 0:
                cosine_similarity[j] = 0
            else:
                cosine_similarity[j] = np.dot(v1, v2) / (v1_norm * v2_norm)

        # compute cosine distance
        cos_dist[i] = 1 - cosine_similarity

    return cos_dist