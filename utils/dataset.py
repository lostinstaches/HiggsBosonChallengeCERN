import csv

import numpy as np


# FEATURES = [Id,Prediction,DER_mass_MMC,DER_mass_transverse_met_lep,DER_mass_vis,DER_pt_h,DER_deltaeta_jet_jet,DER_mass_jet_jet,DER_prodeta_jet_jet,DER_deltar_tau_lep,DER_pt_tot,DER_sum_pt,DER_pt_ratio_lep_tau,DER_met_phi_centrality,DER_lep_eta_centrality,PRI_tau_pt,PRI_tau_eta,PRI_tau_phi,PRI_lep_pt,PRI_lep_eta,PRI_lep_phi,PRI_met,PRI_met_phi,PRI_met_sumet,PRI_jet_num,PRI_jet_leading_pt,PRI_jet_leading_eta,PRI_jet_leading_phi,PRI_jet_subleading_pt,PRI_jet_subleading_eta,PRI_jet_subleading_phi,PRI_jet_all_pt]
# IDXS = []
IDX24_DEPENDENT = [6, 7, 8, 14, 28, 29, 30]
IDX24_DEPENDENT = [e - 2 for e in IDX24_DEPENDENT]
IDX24_DEPENDENT = set(IDX24_DEPENDENT)


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

def clean_dataset(X):
    # TODO: Do something better (e.g. median, mean):
    # X[X == -999.0] = 0.0
    print("CLEANING")
    # undefined -> -999.0
    for r_idx, row in enumerate(X):
        for c_idx, col in enumerate(row):
            if c_idx in IDX24_DEPENDENT and X[r_idx][24 - 2] <= 1:
                X[r_idx][c_idx] = -999.0

    # -999.0 -> mean
    means = np.mean(X, axis=0)
    for r_idx, row in enumerate(X):
        for c_idx, col in enumerate(row):
            if X[r_idx][c_idx] == -999.0:
                X[r_idx][c_idx] = means[c_idx]
    print("CLEANING DONE")
    return X

def normalize_dataset(X):
    # TODO: Again, find something more robust
    X -= np.mean(X, axis=0)
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    div = maxs - mins
    X /= div
    return X

def mean_centering(X):
    mean = X.mean(axis=0)
    return X - mean

def max_division(X):
    print("max is {}".format(np.max(X)))
    return X / np.max(X)

def std_normalization(X):
    return X / np.std(X, axis=0)

def pca_and_whitening(X):
    X = mean_centering(X)
    cov = np.dot(X.T, X) / X.shape[0]
    U,S,V = np.linalg.svd(cov)
    Xrot = np.dot(X, U)
    Xwhite = Xrot / np.sqrt(S + 1e-5)
    # Xwhite = Xwhite[:,:20]
    return Xwhite

def feature_engineering(X):
    PRI_JET_NUM_COL = 22

    pri_jet_num_0 = (X[:,PRI_JET_NUM_COL] == 0).astype(np.float32)
    pri_jet_num_1 = (X[:,PRI_JET_NUM_COL] == 1).astype(np.float32)
    pri_jet_num_2 = (X[:,PRI_JET_NUM_COL] == 2).astype(np.float32)
    pri_jet_num_3 = (X[:,PRI_JET_NUM_COL] == 3).astype(np.float32)

    new_features = [pri_jet_num_0, pri_jet_num_1, pri_jet_num_2, pri_jet_num_3]

    pri_jet_bigger_than_1 = [4, 5, 6, 12, 26, 27, 28]
    for col in pri_jet_bigger_than_1:
        new_feature = X[:,col] * pri_jet_num_2 + X[:,col] * pri_jet_num_3
        new_features.append(new_feature)

    new_features_num = X.shape[1] + len(new_features)
    new_X = np.random.randn(X.shape[0], new_features_num)

    for i in range(len(new_features)):
        new_X[:, X.shape[1]+i] = new_features[i]

    return new_X



def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    new_X = np.zeros((x.shape[0], x.shape[1] * (degree)))
    for i in range(1, degree+1):
        new_X[:,x.shape[1]*(i-1):x.shape[1]*(i)] = x ** i
    return new_X

def preprocess_dataset(X):
    medians = np.median(X, axis=0)
    for r_idx, row in enumerate(X):
        for c_idx, col in enumerate(row):
            if X[r_idx][c_idx] == -999.0:
                X[r_idx][c_idx] = 0.0

    X = build_poly(X, 6)
    X = mean_centering(X)
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
