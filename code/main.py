# basics
import os
import csv
import argparse
import math
import random

# data science
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# our models
from random_forest import RandomForest
from kmeans import Kmeans
from knn import KNN
from naive_bayes import NaiveBayes
from stacking import Stacking


def load_dataset(filename):
    # read csv file
    with open(os.path.join('..', 'data', filename), 'r') as csv_file:
        # read csv data into NumPy array
        data = np.genfromtxt(csv_file, delimiter=',')

        # delete name row
        data = np.delete(data, 0, 0)

        # get dimension of array
        row, col = data.shape

        # split data into X and y
        data = np.hsplit(data, [col - 1])
        out = {
            'X': data[0],
            'y': data[1].flatten().astype(int)
        }

        return out


if __name__ == '__main__':
    # setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True)

    # get --model field of argument
    model = parser.parse_args().model

    if model == 'rf':       # Random Forest
        
        train_data = load_dataset('wordvec_train.csv')
        test_data = load_dataset('wordvec_test.csv')

        # fit and predict datasets
        model = RandomForest(num_trees=15)
        model.create_splits(train_data['X'])
        model.fit(train_data['X'], train_data['y'])

        # compute training error
        y_pred = model.predict(train_data['X'])
        training_error = np.mean(y_pred != train_data['y'])
        print('Random Forest\nTraining error: ' + str(training_error))
        
        # compute test error
        y_pred = model.predict(test_data['X'])
        test_error = np.mean(y_pred != test_data['y'])
        print('Test error: ' + str(test_error))


    elif model == 'nb':     # Naive Bayes

        train_data = load_dataset('wordvec_train.csv')
        test_data = load_dataset('wordvec_test.csv')

        # fit and predict the datasets
        model = NaiveBayes(num_classes=2)
        model.fit(train_data['X'], train_data['y'])

        # compute training error
        y_pred = model.predict(train_data['X'])
        training_error = np.mean(y_pred != train_data['y'])
        print('Naive Bayes\nTraining error: ' + str(training_error))

        # compute test error
        y_pred = model.predict(test_data['X'])
        test_error = np.mean(y_pred != test_data['y'])
        print('Test error: ' + str(test_error))


    elif model == 'st':     # Stacking

        train_data = load_dataset('wordvec_train.csv')
        test_data = load_dataset('wordvec_test.csv')

        # fit and predict the datasets
        model = Stacking()
        model.fit(train_data['X'], train_data['y'])

        # compute training error
        y_pred = model.predict(train_data['X'])
        training_error = np.mean(y_pred != train_data['y'])
        print('Stacking\nTraining error: ' + str(training_error))

        # compute test error
        y_pred = model.predict(test_data['X'])
        test_error = np.mean(y_pred != test_data['y'])
        print('Test error: ' + str(test_error))


    elif model == 'knn':    # K-Nearest Neighbors
        
        train_data = load_dataset('wordvec_train.csv')
        test_data = load_dataset('wordvec_test.csv')

        # fit and predict the datasets
        model = KNN(k=3)
        model.fit(train_data['X'], train_data['y'])

        # compute training error
        y_pred = model.predict(train_data['X'])
        training_error = np.mean(y_pred != train_data['y'])
        print('K-Nearest Neighbors\nTraining error: ' + str(training_error))

        # compute test error
        y_pred = model.predict(test_data['X'])
        test_error = np.mean(y_pred != test_data['y'])
        print('K-Nearest Neighbors\nTest error: ' + str(test_error))

    elif model == 'km':     # K-Means
        
        train_data = load_dataset('wordvec_train.csv')
        test_data = load_dataset('wordvec_test.csv')

        # get a feature (any feature is fine distribution should be the same) and reshape to (n,1)
        X = train_data['X']
        X = X[:, 100]
        X = np.reshape(X, [X.size, 1])

        '''
        Code below helps determine reasonable k for K-Means using Elbow Method
        Result: A reasonable k would be 3
        '''
        # # 1 < k <= 10
        # ks = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # # initialize array of minimum errors at corresponding ks
        # min_errs = np.zeros(len(ks))

        # # compute min error at each k
        # for i in range(len(ks)):
        #     model = Kmeans(k=ks[i])
        #     min_err = np.inf
        #     for itr in range(50):
        #         model.fit(X)
        #         error = model.error(X)
        #         if error < min_err:
        #             min_err = error
        #     min_errs[i] = min_err

        # plt.plot(ks, min_errs, '-ok')
        # plt.xlabel("k")
        # plt.ylabel("Minimum error")

        # fname = os.path.join("..", "figs", "kmeans_min_error_vs_k.png")
        # plt.savefig(fname)
        # print("\nFigure saved as '%s'" % fname)

        '''
        Visualize what the clusters look like at k=3
        '''
        # model = Kmeans(k=3)
        # min_err = np.inf
        # min_err_y = []
        # for i in range(50):
        #     model.fit(X)
        #     error = model.error(X)
        #     if error < min_err:
        #         min_err = error
        #         min_err_y = model.predict(X)

        # print(model.means)
        # print(min_err_y)
        
        # y_axis = np.ones([1, X.size])
        # plt.scatter(X[:,0], y_axis, c=min_err_y, cmap="jet")

        # fname = os.path.join("..", "figs", "kmeans_clustering.png")
        # plt.savefig(fname)
        # print("\nFigure saved as '%s'" % fname)

        
    else:
        print("Invalid model.")