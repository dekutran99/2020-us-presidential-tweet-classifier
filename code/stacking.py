import numpy as np
import utils
from random_forest import RandomForest
from knn import KNN
from naive_bayes import NaiveBayes
from random_forest import DecisionTree

class Stacking():

    def __init__(self):
        pass

    def fit(self, X, y):
        # instantiate the input models
        rf = RandomForest(num_trees=15)
        knn = KNN(k=3)
        nb = NaiveBayes(num_classes=2)

        # Random Forest fit and predict
        rf.create_splits(X)
        rf.fit(X, y)
        rf_pred = rf.predict(X)

        # K-Nearest Neighbors fit and predict
        knn.fit(X, y)
        knn_pred = knn.predict(X)

        # Naive Bayes fit and predict
        nb.fit(X, y)
        nb_pred = nb.predict(X)

        # use predictions from input models as inputs for meta-classifiers
        meta_input = np.hstack((rf_pred.reshape((rf_pred.size, 1)), knn_pred.reshape((knn_pred.size, 1)), nb_pred.reshape((nb_pred.size, 1))))

        # use Decision Tree as meta-classifier
        dt = DecisionTree(max_depth=np.inf)
        dt.fit(meta_input, y)

        self.rf = rf
        self.knn = knn
        self.nb = nb
        self.meta_classifier = dt

    def predict(self, X):
        # instantiate the input models
        rf = self.rf
        knn = self.knn
        nb = self.nb

        # predict using input models
        rf_pred = rf.predict(X)
        knn_pred = knn.predict(X)
        nb_pred = nb.predict(X)

        # use predictions from above 3 models as inputs for meta-classifiers
        meta_input = np.hstack((rf_pred.reshape((rf_pred.size, 1)), knn_pred.reshape((knn_pred.size, 1)), nb_pred.reshape((nb_pred.size, 1))))

        y_pred = self.meta_classifier.predict(meta_input)

        return y_pred