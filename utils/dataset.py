import csv

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


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
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

def load_train_dataset(train_csv):
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
                        Y_train.append(1.0 if feature_value_str == 's' else -1.0)
                    else:
                        X_train_row.append(np.float32(feature_value_str))
            X_train.append(X_train_row)

    return np.array(X_train), np.array(Y_train)
