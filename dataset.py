import csv

import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    with open(data_path, "r") as f:
        debug_l = [x.strip() for x in f.readlines()[0].split(",")]
    #for idx, fname in enumerate(debug_l):
        #print("{} - {}".format(fname, idx))

    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def mean_centering(X):
    mean = X.mean(axis=0)
    return X - mean

def std_normalization(X):
    return X / np.std(X, axis=0)

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    new_X = np.zeros((x.shape[0], x.shape[1] * (degree)))
    for i in range(1, degree+1):
        new_X[:,x.shape[1]*(i-1):x.shape[1]*(i)] = x ** i
    return new_X

def preprocess_dataset(X, poly_features, use_mean_centering=True, use_std_normalization=True):
    medians = np.median(X, axis=0)
    for r_idx, row in enumerate(X):
        for c_idx, col in enumerate(row):
            if X[r_idx][c_idx] == -999.0:
                X[r_idx][c_idx] = 0.0

    X = build_poly(X, poly_features)
    if use_mean_centering:
        X = mean_centering(X)
    if use_std_normalization:
        X = std_normalization(X)

    return X

def delete_features(X, features_to_delete):
    features_to_delete = set(features_to_delete)
    final_features_num = X.shape[1] - len(features_to_delete)
    new_X = np.zeros((X.shape[0], final_features_num), dtype=np.float32)
    cur_f = 0
    for f in range(X.shape[1]):
        if f not in features_to_delete:
            new_X[:, cur_f] = X[:, f]
            cur_f += 1
    return new_X

def keep_features(X, features_to_keep):
    features_to_keep = set(features_to_keep)
    final_features_num = len(features_to_keep)
    new_X = np.zeros((X.shape[0], final_features_num), dtype=np.float32)
    cur_f = 0
    for f in range(X.shape[1]):
        if f in features_to_keep:
            new_X[:, cur_f] = X[:, f]
            cur_f += 1
    return new_X

def split_data(y, x, ratio, seed=1):
    N = y.shape[0]
    N_train = int(N * ratio)

    random_indices = np.random.permutation(N)
    train_indices = random_indices[:N_train]
    val_indices  = random_indices[N_train:]

    return y[train_indices], y[val_indices], x[train_indices], x[val_indices]
