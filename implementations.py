import csv
import os
import pickle
import inspect

import numpy as np


# We use globals to make iterative functions display
# things like validation accuracy conditionally (we
# must use globals since interface of the function is
# fixed in project description PDF)

VERBOSE_OUTPUT = False

X_VALID = None
Y_VALID = None

def set_validation_dataset(x, y):
    """Sets validation dataset to make it available for other functions"""
    global X_VALID
    global Y_VALID
    X_VALID = x
    Y_VALID = y

def set_verbose_output(be_verbose):
    """Enables verbose output"""
    global VERBOSE_OUTPUT
    VERBOSE_OUTPUT = be_verbose

def mse_gradient(y, tx, w):
    """Calculate gradient of MSE loss"""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def calculate_mse(e):
    """Calculate the mse for vector e"""
    return 1/2*np.mean(e**2)

def calculate_mae(e):
    """Calculate the mae for vector e"""
    return np.mean(np.abs(e))

def compute_mse_loss(y, tx, w):
    """Calculate MSE loss"""
    e = y - tx.dot(w)
    return calculate_mse(e)

def numeric_gradient(x, w, y, loss_f, eps=1e-5):
    """Numeric gradient for debugging purposes"""
    grad = np.zeros(w.shape[0], dtype=np.float32)
    for i in range(w.shape[0]):
        w_minus = w.copy()
        w_minus[i] -= eps
        w_plus = w.copy()
        w_plus[i] += eps
        f_x_minus = loss_f(y, x, w_minus)
        f_x_plus = loss_f(y, x, w_plus)
        grad[i] = (f_x_plus - f_x_minus) / (2.0 * eps)
    return grad

def numeric_gradient_diagnostics(y, tx, w, grad, loss_fn):
    """Debug function that verifies gradient implementation"""
    numeric_grad = numeric_gradient(tx, w, y, loss_fn)
    print("Numeric gradient check pass: {}".format(
        np.allclose(grad, numeric_grad)))

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm using mse."""
    w = initial_w
    for n_iter in range(max_iters):
        grad, err = mse_gradient(y, tx, w)
        w -= gamma * grad
        loss = compute_mse_loss(y, tx, w)
        if VERBOSE_OUTPUT:
            print("Gradient Descent({bi}/{ti}): gamma={g} mse-loss={l} ".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w=w, g=gamma))
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent."""

    # For reproductability, normally wouldn't be here
    np.random.seed(1)

    w = initial_w
    N = len(tx)
    for n_iter in range(max_iters):
        random_indices = np.random.permutation(N)
        for num, idx in enumerate(random_indices):
            y_batch, tx_batch = np.array([y[idx]]), np.array([tx[idx]])
            grad, _ = mse_gradient(y_batch, tx_batch, w)
            w -= gamma * grad
            loss = compute_mse_loss(y, tx, w)
            if VERBOSE_OUTPUT:
                print("[Epoch {cur_ep}/{eps}] SGD({bi}/{ti}): loss={l}".format(
                    cur_ep=n_iter, eps=max_iters, bi=num, ti=N, l=loss))
    return w, loss

def least_squares(y, tx):
    A = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    return w, compute_mse_loss(y, tx,  w)

def ridge_regression(y, tx, lambda_):
    N = y.shape[0]
    D = tx.shape[1]
    l = lambda_ * 2.0 * N
    A = tx.T.dot(tx) + l * np.eye(D)
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    return w, compute_mse_loss(y, tx,  w)

def logistic_function(x):
    res = x.copy()
    nonneg = res >= 0
    neg = res < 0
    res[nonneg] = 1 / (1 + np.exp(-res[nonneg]))
    res[neg] = np.exp(res[neg]) / (1 + np.exp(res[neg]))
    return res

def logistic_loss(y, x, w, reg=0.0):
    N = y.shape[0]
    xw = np.dot(x, w)
    log_term = np.log(np.exp(xw) + 1)
    yxw_term = y * xw
    return (1.0 / N) * np.sum(log_term - yxw_term) + (reg / (2 * N)) * np.dot(w, w)

def logistic_gradient(y, x, w, reg=0.0):
    N = y.shape[0]
    probs = logistic_function(np.dot(x, w))
    return (1.0 / N) * np.dot(x.T, probs-y) + (reg / N) * w

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression"""
    w = initial_w
    for n_iter in range(max_iters):
        xw = tx.dot(w)
        preds = logistic_function(xw)
        grad = logistic_gradient(y, tx, w)
        w -= gamma * grad
        if n_iter % 100 == 0 and VERBOSE_OUTPUT:
            print("iter {} loss {}".format(n_iter, logistic_loss(y, tx, w)))
            preds = logistic_function(tx.dot(w))
            preds[preds >=  0.5] = 1.0
            preds[preds < 0.5] = 0.0
            print("Train accuracy {}".format(np.sum(preds == y) / y.shape[0]))
            if X_VALID is not None:
                preds = logistic_function(X_VALID.dot(w))
                preds[preds >=  0.5] = 1.0
                preds[preds < 0.5] = 0.0
                print("Valid accuracy {}".format(np.sum(preds == Y_VALID) / Y_VALID.shape[0]))
    return w, logistic_loss(y, tx, w)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Logistic regression"""
    w = initial_w
    for n_iter in range(max_iters):
        xw = tx.dot(w)
        preds = logistic_function(xw)
        grad = logistic_gradient(y, tx, w, reg=lambda_)
        w -= gamma * grad
        if n_iter % 100 == 0 and VERBOSE_OUTPUT:
            print("iter {} loss {}".format(n_iter, logistic_loss(y, tx, w, reg=lambda_)))
            preds = logistic_function(tx.dot(w))
            preds[preds >=  0.5] = 1.0
            preds[preds < 0.5] = 0.0
            print("Train accuracy {}".format(np.sum(preds == y) / y.shape[0]))
            if X_VALID is not None:
                preds = logistic_function(X_VALID.dot(w))
                preds[preds >=  0.5] = 1.0
                preds[preds < 0.5] = 0.0
                print("Valid accuracy {}".format(np.sum(preds == Y_VALID) / Y_VALID.shape[0]))
    return w, logistic_loss(y, tx, w, reg=lambda_)


# Main training class, inspired by scikit API
class Training(object):
    # Method string -> method fn
    METHOD_NAMES = {
        'least_squares_GD': least_squares_GD,
        'least_squares_SGD': least_squares_SGD,
        'least_squares': least_squares,
        'ridge_regression': ridge_regression,
        'logistic_regression': logistic_regression,
        'reg_logistic_regression': reg_logistic_regression,
    }

    # Methods that use MSE loss
    MSE_LOSS_METHODS = set([
        'least_squares_GD',
        'least_squares_SGD',
        'least_squares',
        'ridge_regression',
    ])
    # Methods that use logistic loss
    LOGISTIC_LOSS_METHODS = set([
        'logistic_regression',
        'reg_logistic_regression'
    ])

    # Sample training_params object (contains training method parameters)
    training_params = {
        'initial_w': None,
        'max_iters': None,
        'gamma': None,
        'max_iters': None,
        'lambda_': None,
    }

    # method_name - method to train parameters on
    # training_params - dict of arguments to function represented by method_name
    def __init__(self, method_name, training_params):
        if method_name not in self.METHOD_NAMES:
            raise ValueError("Wrong method: {}".format(method_name))
        self.method_name = method_name
        # Checks if all params were defined, save them for later
        self._prepare_and_save_training_params(training_params)
        self.w = None
        if VERBOSE_OUTPUT:
            print("-----------------------")
            print("Created training model:")
            print("Method - {}".format(self.method_name))
            print("Params - {}".format(self.params))
            print("-----------------------")

    def _get_function_param_names(self, fn):
        return inspect.getargspec(fn)[0]

    def _prepare_and_save_training_params(self, training_params):
        method_fn = self.METHOD_NAMES[self.method_name]
        method_params = self._get_function_param_names(method_fn)
        for param in method_params:
            if param not in training_params and param != 'y' and param != 'tx':
                raise ValueError("Wrong params: {}".format(training_params))
        params_to_delete = []
        for param in training_params:
            if param not in method_params:
                params_to_delete.append(param)
        for param in params_to_delete:
            del training_params[param]
        self.params = training_params

    # Calculate w based on (X_train, Y_train)
    def fit(self, X_train, Y_train):
        method_fn = self.METHOD_NAMES[self.method_name]
        w, loss = method_fn(Y_train, X_train, **self.params)
        self.w = w
        return w, loss

    # Use trained w to evaluate on validation dataset
    def eval(self, X_valid, Y_valid):
        if self.w is None:
            RuntimeError("eval must be called after fit")
        use_logistic_loss = self.method_name in self.LOGISTIC_LOSS_METHODS
        if use_logistic_loss:
            preds = logistic_function(X_valid.dot(self.w))
            reg = 0.0
            if 'lambda_' in self.params:
                reg = self.params['lambda_']
            loss = logistic_loss(Y_valid, X_valid, self.w, reg)
        else:
            preds = X_valid.dot(self.w)
            loss = compute_mse_loss(Y_valid, X_valid, self.w)
        preds[preds >= 0.5] = 1.0
        preds[preds < 0.5] = 0.0
        acc = np.sum(preds == Y_valid) / Y_valid.shape[0]
        return acc, loss

    # Use trained w to make predictions on test dataset
    def predict(self, X_test):
        if self.w is None:
            RuntimeError("predict must be called after fit")
        use_logistic_loss = self.method_name in self.LOGISTIC_LOSS_METHODS
        if use_logistic_loss:
            Y_test = logistic_function(X_test.dot(self.w))
        else:
            Y_test = X_test.dot(self.w)
        Y_test[Y_test >= 0.5] = 1.0
        Y_test[Y_test < 0.5] = 0.0
        return Y_test

