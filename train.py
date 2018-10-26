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
    # features_to_delete = [14, 17, 18]

    # load the train data
    Y_train, X_train, indexes = dataset.load_csv_data(TRAIN_DATASET)
    # {-1, 1} -> {0, 1}
    Y_train = positive_negative_to_binary(Y_train)

    PRI_JET_NUM_COL = 22
    PRI_JET_NUMS = range(4)

    # pri_jet_num based dataset splitting
    train_datasets = []
    for num in PRI_JET_NUMS:
        indices = X_train[:, PRI_JET_NUM_COL] == num
        train_datasets.append([X_train[indices], Y_train[indices], indices])

    # We'll use it for dataset splitting, we won't need it
    features_to_delete = [PRI_JET_NUM_COL]

    # pre process train data
    for ds in train_datasets:
        ds[0] = dataset.delete_features(ds[0], features_to_delete)
        ds[0] = dataset.preprocess_dataset(ds[0],
            poly_features=8, use_mean_centering=True, use_std_normalization=True)
        ds[0] = add_bias_column(ds[0])

    for idx, ds in enumerate(train_datasets):
        print("Training pri_jet_num {}".format(idx))

        Y_train, Y_validation, X_train, X_validation = \
            dataset.split_data(ds[1], ds[0], ratio_for_splitting)
        implementations.set_validation_dataset(X_validation, Y_validation)

        model = implementations.Training('reg_logistic_regression', {
            'lambda_': 0.0005,
            'initial_w': np.zeros(X_train.shape[1]),
            'max_iters': 5000 if idx == 0 else 15000,
            'gamma': 0.01 if idx == 0 else 0.05,
        })
        model.fit(X_train, Y_train)

        tr_acc, tr_loss = model.eval(X_train, Y_train)
        valid_acc, valid_loss = model.eval(X_validation, Y_validation)

        print("Train loss: {} Train accuracy: {}".format(
            tr_loss, tr_acc))
        print("Validation loss: {} Validation accuracy: {}".format(
            valid_loss, valid_acc))

        ds.append(model)

    # load test data
    Y_test, X_test, indexes = dataset.load_csv_data(TEST_DATA)
    Y_test = positive_negative_to_binary(Y_test)

    test_datasets = []
    for num in PRI_JET_NUMS:
        indices = X_test[:, PRI_JET_NUM_COL] == num
        test_datasets.append([X_test[indices], indices])

    for idx, ds in enumerate(test_datasets):
        print("Testing pri_jet_num {}".format(idx))

        X = dataset.delete_features(ds[0], features_to_delete)
        X = dataset.preprocess_dataset(X,
            poly_features=8, use_mean_centering=True, use_std_normalization=True)
        X = add_bias_column(X)

        model = train_datasets[idx][3]
        Y = model.predict(X)
        Y = binary_to_positive_negative(Y)

        Y_test[ds[1]] = Y

    # create submission .csv file of results Y_test
    dataset.create_csv_submission(indexes, Y_test, "predictions.csv")
