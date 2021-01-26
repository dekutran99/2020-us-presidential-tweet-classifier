
import numpy as np
import utils
from kmeans import Kmeans
from utils import *
from scipy import stats


"""**Global Variables"""


k = 3


"""**Decision Stump**"""


class DecisionStumpErrorRate:

    def __init__(self):
        pass

    def fit(self, X, y):
        N, D = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.splitSat = y_mode
        self.splitNot = None
        self.splitVariable = None
        self.splitValue = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)

        # Loop over features looking for the best split
        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]

                # Find most likely class for each split
                y_sat = mode(y[X[:, d] > value])
                y_not = mode(y[X[:, d] <= value])

                # Make predictions
                y_pred = y_sat * np.ones(N)
                y_pred[X[:, d] <= value] = y_not

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = y_sat
                    self.splitNot = y_not

    def predict(self, X):
        splitVariable = self.splitVariable
        splitValue = self.splitValue
        splitSat = self.splitSat
        splitNot = self.splitNot

        M, D = X.shape

        if splitVariable is None:
            return splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, splitVariable] > splitValue:
                yhat[m] = splitSat
            else:
                yhat[m] = splitNot

        return yhat


"""
helper function that computes the Gini_impurity of the
discrete distribution p.
"""
def Gini_impurity(p):
    # Gini Impurity = G(p_1,...,p_J) = \sum_{i=1}^J p_i * (1 - p_i)
    gini_impurity = 0
    for p_i in p:
        gini_impurity += p_i * (1 - p_i)
    
    return gini_impurity


"""
helper function that computes the Gini Index of a split
"""
def Gini_index(p_l, p_r, N_l, N_r, N_t):
    # Gini Index = \frac{N_l}{N_t}G(p_1^l,...,p_J^l) + \frac{N_r}{N_t}G(p_1^r,...,p^J^r)
    gini_index = (N_l / N_t) * Gini_impurity(p_l) + (N_r / N_t) * Gini_impurity(p_r)
    
    return gini_index

        
class DecisionStumpGiniIndex(DecisionStumpErrorRate):

    def fit(self, X, y, split_features=None, thresholds=None):
        # Get number of samples and features
        N, D = X.shape

        # get count of each class
        class_count = np.bincount(y, minlength=2)
        # get probability of each class
        class_p = class_count / N

        # Compute Gini Impurity
        gini_impurity = Gini_impurity(class_p)

        self.splitVariable = None
        self.splitValue = None
        self.splitSat = np.argmax(class_count)
        self.splitNot = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        # min Gini Index
        minGiniIdx = gini_impurity

        if split_features is None:
            print('Split features not specified.')
            split_features = range(D)

        for d in split_features:
            if thresholds != None:
                # get and reshape feature d (n,)->(n,1) 
                ft = X[:, d]
                ft = np.reshape(ft, [ft.size, 1])

                # get cluster's means for this feature
                means = thresholds[d]
            
                # inititalize K-Means model
                k_means = Kmeans(k=k)
                # get K-Means instance for this feature d
                k_means.means = means

                # assign data in features to cluster
                ft_clusters = k_means.predict(ft)

                # get corresponding cluster's mean for each data
                ft_means = np.ones(ft_clusters.size)
                for i in range(ft_clusters.size):
                    cluster = ft_clusters[i]
                    mean = means[cluster][0]
                    ft_means[i] = mean

                for mean in means:
                    # use clusters' means of this feature as split value
                    r_split = y[ft_means > mean]
                    l_split = y[ft_means <= mean]
                    # get count of labels in each split
                    N_r = r_split.size
                    N_l = l_split.size
                    # get count of each class in each split
                    r_count = np.bincount(r_split)
                    l_count = np.bincount(l_split)
                    # get probability of each class in each split
                    p_r = r_count / N_r
                    p_l = l_count / N_l

                    # compute Gini Index
                    gini_index = Gini_index(p_l, p_r, N_l, N_r, N)
                
                    if gini_index < minGiniIdx:
                        # store this split since it's the current min
                        minGiniIdx = gini_index
                        self.splitVariable = d
                        self.splitValue = mean
                        self.splitSat = mode(r_split)
                        self.splitNot = mode(l_split)
            else:  
                for n in range(N):
                    # get value of feature d at sample n
                    value = X[n, d]

                    # Find class labels for each split
                    r_split = y[X[:, d] > value]
                    l_split = y[X[:, d] <= value]
                    # get count of labels in each split
                    N_r = r_split.size
                    N_l = l_split.size
                    # get count of each class in each split
                    r_count = np.bincount(r_split)
                    l_count = np.bincount(l_split)
                    # get probability of each class in each split
                    p_r = r_count / N_r
                    p_l = l_count / N_l

                    # compute Gini Index
                    gini_index = Gini_index(p_l, p_r, N_l, N_r, N)
                
                    if gini_index < minGiniIdx:
                        # store this split since it's the current min
                        minGiniIdx = gini_index
                        self.splitVariable = d
                        self.splitValue = value
                        self.splitSat = mode(r_split)
                        self.splitNot = mode(l_split)


"""**Decision Tree**"""


class DecisionTree:

    def __init__(self, max_depth, stump_class=DecisionStumpErrorRate):
        self.max_depth = max_depth
        self.stump_class = stump_class

    def fit(self, X, y, thresholds=None):
        # Fits a decision tree using greedy recursive splitting
        N, D = X.shape

        # Learn a decision stump
        splitModel = self.stump_class()
        splitModel.fit(X, y)

        if self.max_depth <= 1 or splitModel.splitVariable is None:
            # If we have reached the maximum depth or the decision stump does
            # nothing, use the decision stump

            self.splitModel = splitModel
            self.subModel1 = None
            self.subModel0 = None
            return

        # Fit a decision tree to each split, decreasing maximum depth by 1
        j = splitModel.splitVariable
        value = splitModel.splitValue

        # Find indices of examples in each split
        splitIndex1 = X[:, j] > value
        splitIndex0 = X[:, j] <= value

        # Fit decision tree to each split
        self.splitModel = splitModel
        self.subModel1 = DecisionTree(self.max_depth - 1, stump_class=self.stump_class)
        self.subModel1.fit(X[splitIndex1], y[splitIndex1], thresholds=thresholds)
        self.subModel0 = DecisionTree(self.max_depth - 1, stump_class=self.stump_class)
        self.subModel0.fit(X[splitIndex0], y[splitIndex0], thresholds=thresholds)

    def predict(self, X):
        M, D = X.shape
        y = np.zeros(M)

        # GET VALUES FROM MODEL
        splitVariable = self.splitModel.splitVariable
        splitValue = self.splitModel.splitValue
        splitSat = self.splitModel.splitSat

        if splitVariable is None:
            # If no further splitting, return the majority label
            y = splitSat * np.ones(M)

        # the case with depth=1, just a single stump.
        elif self.subModel1 is None:
            return self.splitModel.predict(X)

        else:
            # Recurse on both sub-models
            j = splitVariable
            value = splitValue

            splitIndex1 = X[:, j] > value
            splitIndex0 = X[:, j] <= value

            y[splitIndex1] = self.subModel1.predict(X[splitIndex1])
            y[splitIndex0] = self.subModel0.predict(X[splitIndex0])

        return y

class RandomStumpGiniIndex(DecisionStumpGiniIndex):

        def fit(self, X, y, thresholds=None):
            # Randomly select k features.
            # This can be done by randomly permuting
            # the feature indices and taking the first k
            D = X.shape[1]
            k = int(np.floor(np.sqrt(D)))

            chosen_features = np.random.choice(D, k, replace=False)

            DecisionStumpGiniIndex.fit(self, X, y, split_features=chosen_features, thresholds=thresholds)


"""**Random Tree**"""


class RandomTree(DecisionTree):

    def __init__(self, max_depth):
        DecisionTree.__init__(self, max_depth=max_depth, stump_class=RandomStumpGiniIndex)

    def fit(self, X, y, thresholds=None):
        N = X.shape[0]
        boostrap_inds = np.random.choice(N, N, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]

        DecisionTree.fit(self, bootstrap_X, bootstrap_y, thresholds=thresholds)


"""**Random Forest**"""


class RandomForest:

    def __init__(self, num_trees, max_depth=np.inf):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.thresholds = None

    def fit(self, X, y):
        self.trees = []
        for m in range(self.num_trees):
            tree = RandomTree(max_depth=self.max_depth)
            tree.fit(X, y, thresholds=self.thresholds)
            self.trees.append(tree)

    def predict(self, X):
        t = X.shape[0]
        yhats = np.ones((t, self.num_trees), dtype=np.uint8)
        # Predict using each model
        for m in range(self.num_trees):
            yhats[:, m] = self.trees[m].predict(X)

        # Take the most common label
        return stats.mode(yhats, axis=1)[0].flatten()

    def create_splits(self, X):
        # get shape of dataset
        N, D = X.shape

        # thresholds is set of K-Means of each feature
        self.thresholds = []

        for d in range(D):
            # reshape (n,) to (n,1)
            feature = X[:, d]
            feature = np.reshape(feature, [feature.size, 1])

            # Initialize K-Means model
            k_means = Kmeans(k=k)
            min_err = np.inf
            min_err_means = None

            for i in range(50):
                k_means.fit(feature)
                error = k_means.error(feature)
                if error < min_err:
                    min_err = error
                    min_err_means = k_means.means

            self.thresholds.append(min_err_means)