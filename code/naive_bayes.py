import numpy as np
import math

class NaiveBayes:

    def __init__(self, num_classes, beta=1):
        self.num_classes = num_classes
        self.beta = beta

    def fit(self, X, y):
        # get number of examples and features
        N, D = X.shape

        # number of classes
        C = self.num_classes

        # number of examples per class
        n_cs = np.bincount(y)
        # probability of each class
        p_y = n_cs / N

        # distribution parameters for each class at each feature
        means = np.zeros([D, C])
        vars = np.zeros([D, C])

        # compute means and variances of each class at each feature
        for d in range(D):
            for c in range(C):
                # mean of feature d with class c
                mean_dc = (1 / n_cs[c]) * np.sum(X[y == c][:, d])
                # var of feature d with class c
                var_dc = (1 / n_cs[c]) * np.sum((mean_dc - X[y == c][:, d])**2)

                means[d, c] = mean_dc
                vars[d, c] = var_dc

        # save prediction parameters
        self.p_y = p_y
        self.means = means
        self.vars = vars

    def predict(self, X):
        N, D = X.shape
        C = self.num_classes

        # get prior class probabilities, Gaussian distribution paremeter
        p_y = self.p_y
        means = self.means
        vars = self.vars
        
        '''
        the product of small probailities goes to zero very fast
        => we compute p(x_i | y_i=y_c) in log space
        '''
        ln_p_xy = np.zeros([N, C])
        for n in range(N):
            for d in range(D):
                x_nd = X[n, d]
                for c in range(C):
                    ln_p_xy[n, c] += (1 / 2) * (((x_nd - means[d, c]) / math.sqrt(vars[d, c]))**2) + math.log(math.sqrt(vars[d, c]) * math.sqrt(2*math.pi))

        # make predictions based on lowest class probability
        y_pred = np.zeros(N)
        for n in range(N):
            y_pred[n] = np.argmin(np.log(p_y) + ln_p_xy[n, :])

        return y_pred