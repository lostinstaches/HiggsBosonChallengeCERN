import csv
import os
import pickle

import numpy as np


X_VALID = None
Y_VALID = None


def set_valid(x, y):
    global X_VALID
    global Y_VALID
    X_VALID = x
    Y_VALID = y

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

# y  - target data
# tx - independend data
# initial_w - vektor of parameters to optimize ([w0, w1, ..])
# max_iters - stepsize of optimization 
# gamma     - learning rate )
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm using mse."""
    # Define parameters to store w's for each step for later visualization
    ws = [initial_w]
    # Define parameter to store all mse losses for each step for later visualization
    losses = []
    # first w values (here [0,0])
    w = initial_w
    # optimization loop
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = mse_gradient(y, tx, w)
        # compute single value mse of all errors err (is a vector of each error for each datapoint)
        curr_mse_loss = calculate_mse(err)
        # gradient w's by descent update all parameters
        w = w - gamma * grad
        # store w 
        ws = w
        # store loss
        losses = curr_mse_loss
        print("Gradient Descent({bi}/{ti}): gamma={g} mse-loss={l} ".format(bi=n_iter, ti=max_iters - 1, l=curr_mse_loss, w=w, g=gamma))
    return losses, ws

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    # optimization loop
    for n_iter in range(max_iters):
        # choose random batch sample set (size according to batch_size)
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss for the random sample batches (n = batch_size)
            # loss gets ignored here because this is only a sample set of the whole data
            grad, _ = mse_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss over all data
            loss = compute_loss(y, tx, w)
            # store w and loss
            #ws.append(w)
            #losses.append(loss)
            ws = w

        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

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
            # print("w {}".format(w))
    return w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Logistic regression"""
    w = initial_w
    for n_iter in range(max_iters):
        xw = tx.dot(w)
        preds = logistic_function(xw)
        grad = logistic_gradient(tx, w, y, reg=lambda_)
        #numeric_grad = numeric_gradient(tx_batch, w, y_batch, logistic_loss)
        #print("grad {}".format(grad))
        #print("numeric_grad {}".format(numeric_gradient(tx_batch, w, y_batch, logistic_loss)))
        #print("both subtracted {}".format(grad - numeric_grad))
        w -= gamma * grad
        if n_iter % 100 == 0:
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
    return w

t = Training('least_squares_GD', params)
t.fit(X_train, y_train)
predictions = t.predict(X_test)


class Training(object):
	method_names = ['least_squares_GD', 'least_squares_SGD', 'least_squares', 'ridge_regression', 'logistic_regression', 'reg_logistic_regression']
	params = {
		'initial_w': None,
		'max_iters': None,
		'gamma': None,
		'max_iters': None,
		'lambda_': None,
		'poly_features': 6,
		'mean_subtraction': True,
		'std_division': True,
		'verbose': True,
	}

	def __init__(self, method_name, params):
		pass

	def fit(self, X_train, y_train):
		if 'verbose':
			ekwoakeoawk
		return w, loss

	def predict(self, X_test):
		return Y_test

