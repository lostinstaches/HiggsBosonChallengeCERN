#!/usr/bin/env python

import os

from utils import dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATASET = os.path.join(SCRIPT_DIR, 'data/train.csv')
CACHE_DIR = os.path.join(SCRIPT_DIR, 'cache')


if __name__ == '__main__':
    # Y_train is array of 0/1 values based on "Prediction" feature (s -> 1, b -> 0)
    # we're ignoring "Id" feature
    X_train, Y_train = dataset.load_train_dataset(TRAIN_DATASET, CACHE_DIR)
    print(X_train[:10])
    print(Y_train[:10])

    # X_train, Y_train = load_train_dataset(TRAIN_DATASET) NxD
    # X_train, Y_train = clean_dataset("METHOD_NAME", X_train, Y_train)
    # w = logistic_regression(X_train, Y_train)
    # TESTING TIME using w

