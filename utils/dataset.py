import csv
import os
import pickle

import numpy as np


class FeatureMetadata(object):
    def __init__(self):
        self.feature_idx = -1
        self.type = np.float32

    def __repr__(self):
        s = "FeatureMetadata(\n"
        s += "\tfeature_idx: {}".format(self.feature_idx)
        s += "\n\ttype: {}".format(self.type)
        s += "\n)"
        return s

    def string_to_number(self, s):
        return np.float32(s)


def load_cached_train_dataset(X_train_pkl, Y_train_pkl):
    with open(X_train_pkl, 'rb') as f:
        X_train = pickle.load(f)
    with open(Y_train_pkl, 'rb') as f:
        Y_train = pickle.load(f)
    return X_train, Y_train

def cache_train_dataset(X_train, X_train_pkl, Y_train, Y_train_pkl):
    with open(X_train_pkl, 'wb') as f:
        pickle.dump(X_train, f)
    with open(Y_train_pkl, 'wb') as f:
        pickle.dump(Y_train, f)


def load_train_dataset(train_csv, cache_dir):
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    X_train_pkl = os.path.join(cache_dir, 'X_train.pkl')
    Y_train_pkl = os.path.join(cache_dir, 'Y_train.pkl')
    if os.path.exists(X_train_pkl) and os.path.exists(Y_train_pkl):
        return load_cached_train_dataset(X_train_pkl, Y_train_pkl)

    X_train = []
    Y_train = []

    features_metadata = {}
    idx_to_feature = {}
    with open(train_csv) as csvfile:
        reader = csv.reader(csvfile)
        for idx, feature in enumerate(next(reader)):
            if feature == 'Id':
                continue
            idx_to_feature[idx] = feature
            features_metadata[feature] = FeatureMetadata()
            features_metadata[feature].feature_idx = idx

        for row in reader:
            X_train_row = []
            for idx, feature_value_str in enumerate(row):
                if idx in idx_to_feature:
                    if idx_to_feature[idx] == 'Prediction':
                        Y_train.append(1.0 if feature_value_str == 's' else 0.0)
                    else:
                        X_train_row.append(np.float32(feature_value_str))
            X_train.append(X_train_row)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    cache_train_dataset(X_train, X_train_pkl, Y_train, Y_train_pkl)

    return np.array(X_train), np.array(Y_train)
