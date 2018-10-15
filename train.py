#!/usr/bin/env python

from utils import dataset
from utils import helpers

import numpy as np

TRAIN_DATASET = 'data/train.csv'
TEST_DATA = 'data/test.csv'



if __name__ == '__main__':
    # Y_train is array of 0/1 values based on "Prediction" feature (s -> 1, b -> 0)
    # we're ignoring "Id" feature

    lamda_ = 5
    ratio_for_splitting = 0.8


    Y_train, X_train, indexes = dataset.load_csv_data(TRAIN_DATASET)

    # split_data
    Y_train, Y_validation, X_train, X_validation = dataset.split_data(Y_train, X_train, ratio_for_splitting)

    # pre process data
    X_train = dataset.preprocess_dataset(X_train)

    # ridge regression
    w = helpers.ridge_regression(Y_train, X_train, lamda_)
    # training error
    loss_train = helpers.compute_loss(Y_train, X_train, w)
    print("Loss for train data is: ", loss_train)
    # validation error
    loss_val = helpers.compute_loss(Y_validation, X_validation, w)
    print("Loss for validation data is: ", loss_val)


    Y_test, X_test, indexes = dataset.load_csv_data(TEST_DATA)
    X_test = dataset.preprocess_dataset(X_test)
    Y_test = X_test.dot(w)

    Y_test = [1 if e >= 0.0 else -1 for e in Y_test]

    # create submission file of results
    dataset.create_csv_submission(indexes, Y_test, "dot.csv")
