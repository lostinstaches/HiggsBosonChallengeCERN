#!/usr/bin/env python

import dataset
import implementations

import numpy as np

TRAIN_DATASET = 'data/train.csv'
TEST_DATA = 'data/test.csv'


def add_bias_column(X):
    X_temp = np.ones((X.shape[0], X.shape[1]+1))
    X_temp[:,:-1] = X
    X = X_temp
    return X

def positive_negative_to_binary(Y):
    Y[Y == -1.0] = 0.0
    return Y

def binary_to_positive_negative(Y):
    Y[Y == 0.0] = -1.0
    return Y

if __name__ == '__main__':
    lambda_ = 5
    ratio_for_splitting = 0.90
    implementations.set_verbose_output(True)

    # chosen experimenttally
    features_to_delete = [14, 17, 18]

    # load the train data
    Y_train, X_train, indexes = dataset.load_csv_data(TRAIN_DATASET)
    # {-1, 1} -> {0, 1}
    Y_train = positive_negative_to_binary(Y_train)

    # pre process train data
    X_train = dataset.delete_features(X_train, features_to_delete)
    X_train = dataset.preprocess_dataset(X_train,
        poly_features=10, use_mean_centering=True, use_std_normalization=True)
    X_train = add_bias_column(X_train)

    # split_data
    Y_train, Y_validation, X_train, X_validation = \
        dataset.split_data(Y_train, X_train, ratio_for_splitting)
    implementations.set_validation_dataset(X_validation, Y_validation)

    training = implementations.Training('reg_logistic_regression', {
        'lambda_': 0.003,
        'initial_w': np.zeros(X_train.shape[1]),
        'max_iters': 500,
        'gamma': 0.01,
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
    Y_test = positive_negative_to_binary(Y_test)

    # pre process test data
    X_test = dataset.delete_features(X_test, features_to_delete)
    X_test = dataset.preprocess_dataset(X_test,
        poly_features=10, use_mean_centering=True, use_std_normalization=True)
    X_test = add_bias_column(X_test)

    # build results
    Y_test = training.predict(X_test)
    Y_test = binary_to_positive_negative(Y_test)

    # create submission .csv file of results Y_test
    dataset.create_csv_submission(indexes, Y_test, "predictions.csv")
