# -*- coding: utf-8 -*-

import numpy as np

def least_squares_GD():
    pass

def least_squares_SGD():
    pass

def least_squares(y, tx):
    #normal equations
    gram_matrix = np.transpose(tx).dot(tx)
    gram_matrix_inverse = np.linalg.inv(gram_matrix)
    answer = gram_matrix_inverse.dot(np.transpose(tx))
    answer = answer.dot(y)
    return answer


def ridge_regression(y, tx, lamb):
    aI = lamb * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)


def logistic_regression():
    pass

def reg_logistic_regression():
    pass


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)
    # return calculate_mae(e)
