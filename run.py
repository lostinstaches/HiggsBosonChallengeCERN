#!/usr/bin/env python

import dataset
import implementations

import numpy as np

TRAIN_DATASET = 'data/train.csv'
TEST_DATA = 'data/test.csv'

PRI_JET_NUM_SPLIT = True


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
    ratio_for_splitting = 0.99
    implementations.set_verbose_output(True)

    # Load the train data
    Y_train, X_train, indexes = dataset.load_csv_data(TRAIN_DATASET)
    # {-1, 1} -> {0, 1}
    Y_train = positive_negative_to_binary(Y_train)

    PRI_JET_NUM_COL = 22
    PRI_JET_NUMS = range(4)
    train_datasets = []
    features_to_delete = []

    if PRI_JET_NUM_SPLIT:
        # pri_jet_num based dataset splitting
        for num in PRI_JET_NUMS:
            indices = X_train[:, PRI_JET_NUM_COL] == num
            train_datasets.append([X_train[indices], Y_train[indices], indices])
        # pri_jet_num not needed after dataset splitting
        features_to_delete = [PRI_JET_NUM_COL]
    else:
        train_datasets.append([X_train, Y_train, list(range(len(X_train)))])

    # Preprocess train data
    for ds in train_datasets:
        ds[0] = dataset.delete_features(ds[0], features_to_delete)
        ds[0] = dataset.preprocess_dataset(ds[0],
            poly_features=8, use_mean_centering=True, use_std_normalization=True)
        ds[0] = add_bias_column(ds[0])

    # Training code
    for idx, ds in enumerate(train_datasets):
        print("Testing dataset num {}".format(idx))

        Y_train, Y_validation, X_train, X_validation = \
            dataset.split_data(ds[1], ds[0], ratio_for_splitting)
        implementations.set_validation_dataset(X_validation, Y_validation)

        model = implementations.Training('reg_logistic_regression', {
            'lambda_': 0.005,
            'initial_w': np.zeros(X_train.shape[1]),
            'max_iters': 10000,
            'gamma': 0.17,
        })
        model.fit(X_train, Y_train)

        tr_acc, tr_loss = model.eval(X_train, Y_train)
        valid_acc, valid_loss = model.eval(X_validation, Y_validation)

        print("Train loss: {} Train accuracy: {} Total samples: {}".format(
            tr_loss, tr_acc, len(X_train)))
        print("Validation loss: {} Validation accuracy: {} Total samples: {}".format(
            valid_loss, valid_acc, len(X_validation)))

        ds.append(model)

    # Load test data
    Y_test, X_test, indexes = dataset.load_csv_data(TEST_DATA)
    Y_test = positive_negative_to_binary(Y_test)

    test_datasets = []

    if PRI_JET_NUM_SPLIT:
        for num in PRI_JET_NUMS:
            indices = X_test[:, PRI_JET_NUM_COL] == num
            test_datasets.append([X_test[indices], indices])
    else:
        test_datasets.append([X_test, list(range(len(X_test)))])

    # Evaluation code
    for idx, ds in enumerate(test_datasets):
        print("Testing dataset num {}".format(idx))

        X = dataset.delete_features(ds[0], features_to_delete)
        X = dataset.preprocess_dataset(X,
            poly_features=8, use_mean_centering=True, use_std_normalization=True)
        X = add_bias_column(X)

        model = train_datasets[idx][3]
        Y = model.predict(X)
        Y = binary_to_positive_negative(Y)

        Y_test[ds[1]] = Y

    # Create submission .csv file of results Y_test
    dataset.create_csv_submission(indexes, Y_test, "predictions.csv")
