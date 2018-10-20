#!/usr/bin/env python

from utils import dataset
from utils import helpers

import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing

TRAIN_DATASET = 'data/train.csv'
TEST_DATA = 'data/test.csv'

def add_bias_column(X):
    X_temp = np.ones((X.shape[0], X.shape[1]+1))
    X_temp[:,:-1] = X
    X = X_temp
    return X

if __name__ == '__main__':
    # Y_train is array of 0/1 values based on "Prediction" feature (s -> 1, b -> 0)
    # we're ignoring "Id" feature

    lamda_ = 5
    ratio_for_splitting = 0.8

    # load the train data
    Y_train, X_train, indexes = dataset.load_csv_data(TRAIN_DATASET)
    Y_train[np.where(Y_train==-1)] = 0.0
    print(X_train.shape)

    # pre process train data
    X_train = dataset.preprocess_dataset(X_train)

    # Add bias
    X_train = add_bias_column(X_train)
    # X_train = preprocessing.normalize(X_train, norm='l2')
    # X_train = poly.fit_transform(X_train)

    # split_data
    Y_train, Y_validation, X_train, X_validation = dataset.split_data(Y_train, X_train, ratio_for_splitting)
    print(Y_train.shape)
    print(Y_validation.shape)

    # _features = X_train.shape[1]
    # w_initial = np.zeros((_features), dtype=int)
    # max_iters = 5000

    # just uncomment a block for a specific type
    
    # print("Mean Squared Error Gradient Descent")
    # best setting:max_iters 5000  Gradient Descent(14999/14999): gamma=7.4e-07 mse-loss=0.35792074222770376 
    # gamma_MSEGD = 0.00000074
    # w is the last optimized vector of the algorithm (len(w)==30)
    # gradient_losses, w = helpers.least_squares_GD(Y_train, X_train, w_initial, max_iters, gamma_MSEGD) 
    

    '''
    print("Stochastic Mean Squared Error Gradient Descent")
    # best setting even with batch size = every row .. SGD(4999/4999): loss=0.36089722718771117 super costly
    gamma_SMSEGD = 0.00000074
    batch_size = 100000000
    sgd_losses, w = helpers.least_squares_SGD(Y_train, X_train, w_initial, batch_size, max_iters, gamma_SMSEGD)
    '''

    '''
    print("RIDGE")
    # ridge regression
    w = helpers.ridge_regression(Y_train, X_train, lamda_)

    # training error
    loss_train = helpers.compute_loss(Y_train, X_train, w)
    print("Loss for train data is: ", loss_train)

    # validation error
    loss_val = helpers.compute_loss(Y_validation, X_validation, w)
    print("Loss for validation data is: ", loss_val)
    '''
    print(Y_train[:10])

    w = helpers.logistic_regression(Y_train, X_train, np.zeros(X_train.shape[1]), 1000, 0.005)
    print("changing learning rate")
    w = helpers.logistic_regression(Y_train, X_train, w, 1000, 0.001)
    print("changing learning rate")
    w = helpers.logistic_regression(Y_train, X_train, w, 1000, 0.0005)
    print("changing learning rate")
    w = helpers.logistic_regression(Y_train, X_train, w, 1000, 0.0001)
    valid_preds = helpers.logistic_function(X_validation.dot(w))
    valid_preds[valid_preds >=  0.5] = 1.0
    valid_preds[valid_preds < 0.5] = 0.0
    print("Valid accuracy {}".format(np.sum(valid_preds == Y_validation) / Y_validation.shape[0]))

    #clf = SGDClassifier(loss='log', alpha=0.00005, fit_intercept=True, max_iter=1000, verbose=1, n_iter_no_change=100)
    # clf = LogisticRegression(solver='newton-cg', C=1000.0, multi_class='ovr', verbose=1, max_iter=1000)
    # clf.fit(X_train, Y_train)

    # load test data
    Y_test, X_test, indexes = dataset.load_csv_data(TEST_DATA)

    # pre process test data
    X_test = dataset.preprocess_dataset(X_test)
    X_test = add_bias_column(X_test)
    # X_test = preprocessing.normalize(X_test, norm='l2')
    # X_test = poly.fit_transform(X_test)

    # build results
    # Y_test = X_test.dot(w)
    Y_test = helpers.logistic_function(np.dot(X_test, w))
    Y_test[np.where(Y_test >= 0.5)] = 1.0
    Y_test[np.where(Y_test < 0.5)] = -1.0

    # parse results
    # Y_test = [1 if e >= 0.0 else -1 for e in Y_test]
    # Y_test = [1 if e >= 0.5 else -1 for e in Y_test]

    # create submission .csv file of results Y_test
    dataset.create_csv_submission(indexes, Y_test, "dot.csv")
