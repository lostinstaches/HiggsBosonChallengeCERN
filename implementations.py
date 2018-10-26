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
    global X_VALID
    global Y_VALID
    X_VALID = x
    Y_VALID = y

def set_verbose_output(be_verbose):
    global VERBOSE_OUTPUT
    VERBOSE_OUTPUT = be_verbose


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    # Efficient implementation of SGD
    if batch_size == 1 and num_batches == 1:
        ind = np.random.randint(0, data_size)
        chosen_y = np.array([y[ind]])
        chosen_x = tx[ind]
        chosen_x = chosen_x[np.newaxis, :]
        yield chosen_y, chosen_x
    
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def mse_gradient(y, tx, w):
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_mse_loss(y, tx, w):
    e = y - tx.dot(w)
    return calculate_mse(e)

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
    w = initial_w
    for n_iter in range(max_iters):
        # Choose one random sample from dataset
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad, _ = mse_gradient(y_batch, tx_batch, w)
            w -= gamma * grad
            loss = compute_mse_loss(y, tx, w)
        if VERBOSE_OUTPUT:
            print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
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

def logistic_function(z):
    return np.exp(z) / (1 + np.exp(z))

def logistic_loss(x, w, y, reg=0.0):
    N = y.shape[0]
    xw = np.dot(x, w)
    log_term = np.log(np.exp(xw) + 1)
    yxw_term = y * xw
    return (1.0 / N) * np.sum(log_term - yxw_term) + (reg / (2 * N)) * np.dot(w, w)

def logistic_gradient(x, w, y, reg=0.0):
    N = y.shape[0]
    probs = logistic_function(np.dot(x, w))
    return (1.0 / N) * np.dot(x.T, probs-y) + (reg / N) * w

def numeric_gradient(x, w, y, loss_f, eps=1e-5):
    grad = np.zeros(w.shape[0], dtype=np.float32)
    for i in range(w.shape[0]):
        w_minus = w.copy()
        w_minus[i] -= eps
        w_plus = w.copy()
        w_plus[i] += eps
        f_x_minus = loss_f(x, w_minus, y)
        f_x_plus = loss_f(x, w_plus, y)
        grad[i] = (f_x_plus - f_x_minus) / (2.0 * eps)
    return grad

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression"""
    w = initial_w
    for n_iter in range(max_iters):
        xw = tx.dot(w)
        preds = logistic_function(xw)
        grad = logistic_gradient(tx, w, y)
        #print(grad)
        #numeric_grad = numeric_gradient(tx_batch, w, y_batch, logistic_loss)
        #print("grad {}".format(grad))
        #print("numeric_grad {}".format(numeric_gradient(tx_batch, w, y_batch, logistic_loss)))
        #print("both subtracted {}".format(grad - numeric_grad))
        w -= gamma * grad
        if n_iter % 100 == 0:
            print("iter {} loss {}".format(n_iter, logistic_loss(tx, w, y)))
            preds = logistic_function(tx.dot(w))
            preds[preds >=  0.5] = 1.0
            preds[preds < 0.5] = 0.0
            print("Train accuracy {}".format(np.sum(preds == y) / y.shape[0]))
            if X_VALID is not None:
                preds = logistic_function(X_VALID.dot(w))
                preds[preds >=  0.5] = 1.0
                preds[preds < 0.5] = 0.0
                print("Valid accuracy {}".format(np.sum(preds == Y_VALID) / Y_VALID.shape[0]))
    return w, logistic_loss(tx, w, y)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Logistic regression"""
    w = initial_w
    for n_iter in range(max_iters):
        xw = tx.dot(w)
        preds = logistic_function(xw)
        grad = logistic_gradient(tx, w, y, reg=lambda_)
        w -= gamma * grad
        if n_iter % 100 == 0 and VERBOSE_OUTPUT:
            print("iter {} loss {}".format(n_iter, logistic_loss(tx, w, y, reg=lambda_)))
            preds = logistic_function(tx.dot(w))
            preds[preds >=  0.5] = 1.0
            preds[preds < 0.5] = 0.0
            print("Train accuracy {}".format(np.sum(preds == y) / y.shape[0]))
            if X_VALID is not None:
                preds = logistic_function(X_VALID.dot(w))
                preds[preds >=  0.5] = 1.0
                preds[preds < 0.5] = 0.0
                print("Valid accuracy {}".format(np.sum(preds == Y_VALID) / Y_VALID.shape[0]))
    return w, logistic_loss(tx, w, y, reg=lambda_)


class Training(object):
    METHOD_NAMES = {
        'least_squares_GD': least_squares_GD,
        'least_squares_SGD': least_squares_SGD,
        'least_squares': least_squares,
        'ridge_regression': ridge_regression,
        'logistic_regression': logistic_regression,
        'reg_logistic_regression': reg_logistic_regression,
    }

    MSE_LOSS_METHODS = set([
        'least_squares_GD',
        'least_squares_SGD',
        'least_squares',
        'ridge_regression',
    ])
    LOGISTIC_LOSS_METHODS = set([
        'logistic_regression',
        'reg_logistic_regression'
    ])

    training_params = {
        'initial_w': None,
        'max_iters': None,
        'gamma': None,
        'max_iters': None,
        'lambda_': None,
    }

    def __init__(self, method_name, training_params):
        if method_name not in self.METHOD_NAMES:
            raise ValueError("Wrong method: {}".format(method_name))
        self.method_name = method_name
        # Checks if all params were defined, save them for later
        self.prepare_and_save_training_params(training_params)
        self.w = None

    def _get_function_param_names(self, fn):
        return inspect.getargspec(fn)[0]

    def prepare_and_save_training_params(self, training_params):
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

    def fit(self, X_train, Y_train):
        method_fn = self.METHOD_NAMES[self.method_name]
        w, loss = method_fn(Y_train, X_train, **self.params)
        self.w = w
        return w, loss

    def eval(self, X_valid, Y_valid):
        if self.w is None:
            RuntimeError("eval must be called after fit")
        use_logistic_loss = self.method_name in self.LOGISTIC_LOSS_METHODS
        if use_logistic_loss:
            preds = logistic_function(X_valid.dot(self.w))
            reg = 0.0
            if 'lambda_' in self.params:
                reg = self.params['lambda_']
            loss = logistic_loss(X_valid, self.w, Y_valid, reg)
        else:
            preds = X_valid.dot(self.w)
            loss = compute_mse_loss(Y_valid, X_valid, self.w)
        preds[preds >= 0.5] = 1.0
        preds[preds < 0.5] = 0.0
        acc = np.sum(preds == Y_valid) / Y_valid.shape[0]
        return acc, loss

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

