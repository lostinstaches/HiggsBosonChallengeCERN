#!/usr/bin/env python

from utils import dataset
from utils import helpers

import implementations

import numpy as np

TRAIN_DATASET = 'data/train.csv'
TEST_DATA = 'data/test.csv'


def add_bias_column(X):
    X_temp = np.ones((X.shape[0], X.shape[1]+1))
    X_temp[:,:-1] = X
    X = X_temp
    return X

def msegc(Y_train, X_train, w_initial, max_iters, gamma_MSEGD=0.00000074):
    print("Mean Squared Error Gradient Descent")
    # best setting:max_iters 5000  Gradient Descent(14999/14999): gamma=7.4e-07 mse-loss=0.35792074222770376
    #gamma_MSEGD = 0.00000074
    # w is the last optimized vector of the algorithm (len(w)==30)
    gradient_losses, w = helpers.least_squares_GD(Y_train, X_train, w_initial, max_iters, gamma_MSEGD)
    return gradient_losses, w

def smsegd(Y_train, X_train, w_initial, max_iters, batch_size=100000000, gamma_MSEGD=0.00000074):
    print("Stochastic Mean Squared Error Gradient Descent")
    # best setting even with batch size = every row .. SGD(4999/4999): loss=0.36089722718771117 super costly
    #gamma_SMSEGD = 0.00000074
    #batch_size = 100000000
    sgd_losses, w = helpers.least_squares_SGD(Y_train, X_train, w_initial, batch_size, max_iters, gamma_SMSEGD)
    return sgd_losses, w

def ridge_regr():
    print("Ridge regression")
    w = helpers.ridge_regression(Y_train, X_train, lamda_)
    return w

def positive_negative_to_binary(Y):
    Y[Y == -1.0] = 0.0
    return Y

def binary_to_positive_negative(Y):
    Y[Y == 0.0] = -1.0
    return Y

if __name__ == '__main__':
    lambda_ = 5
    ratio_for_splitting = 0.90

    # chosen experimenttally
    features_to_delete = [14, 17, 18]

    # load the train data
    Y_train, X_train, indexes = dataset.load_csv_data(TRAIN_DATASET)
    # {-1, 1} -> {0, 1}
    Y_train = positive_negative_to_binary(Y_train)

    # pre process train data
    X_train = dataset.delete_features(X_train, features_to_delete)
    X_train = dataset.preprocess_dataset(X_train, 10)
    X_train = add_bias_column(X_train)

    # split_data
    Y_train, Y_validation, X_train, X_validation = dataset.split_data(Y_train, X_train, ratio_for_splitting)
    helpers.set_validation_dataset(X_validation, Y_validation)

    training = implementations.Training('ridge_regression', {
        'lambda_': 0.003
    })
    training.fit(X_train, Y_train)

    tr_acc, tr_loss = training.eval(X_train, Y_train)
    valid_acc, valid_loss = training.eval(X_validation, Y_validation)

    print("Train loss: {} Train accuracy: {}".format(
        tr_loss, tr_acc))
    print("Validation loss: {} Validation accuracy: {}".format(
        valid_loss, valid_acc))

    # load test data
    Y_test, X_test, indexes = dataset.load_csv_data(TEST_DATA)
    Y_test[np.where(Y_test==-1)] = 0.0

    # pre process test data
    X_test = dataset.delete_features(X_test, features_to_delete)
    X_test = dataset.preprocess_dataset(X_test, 10)
    X_test = add_bias_column(X_test)

    # build results
    Y_test = training.predict(X_test)
    Y_test = binary_to_positive_negative(Y_test)

    # create submission .csv file of results Y_test
    dataset.create_csv_submission(indexes, Y_test, "predictions.csv")
