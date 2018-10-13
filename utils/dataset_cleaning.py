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


def load_train_dataset(train_csv):
    features_metadata = {}
    with open(train_csv) as csvfile:
        reader = csv.reader(csvfile)
        for idx, feature in enumerate(next(reader)):
            features_metadata[feature] = FeatureMetadata()
            features_metadata[feature].feature_idx = idx
    return features_metadata
