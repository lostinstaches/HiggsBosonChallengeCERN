#!/usr/bin/env python

from utils import dataset

TRAIN_DATASET = 'data/train.csv'


if __name__ == '__main__':
    # Y_train is array of 0/1 values based on "Prediction" feature (s -> 1, b -> 0)
    # we're ignoring "Id" feature

    X_train, Y_train = dataset.load_train_dataset(TRAIN_DATASET)
    print(X_train[:10])
    print(Y_train[:10])

    # X_train, Y_train = load_train_dataset(TRAIN_DATASET) NxD
    # X_train, Y_train = clean_dataset("METHOD_NAME", X_train, Y_train)
    # w = logistic_regression(X_train, Y_train)
    # TESTING TIME using w

