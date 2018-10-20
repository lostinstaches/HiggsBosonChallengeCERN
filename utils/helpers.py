# -*- coding: utf-8 -*-

import numpy as np

def compute_gradient(y, tx, w):
    loss = y - tx.dot(w)
    grad = -tx.T.dot(loss) / len(loss)
    return grad, loss

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
        grad, err = compute_gradient(y, tx, w)
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
            grad, _ = compute_gradient(y_batch, tx_batch, w)
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

def logistic_loss(x, w, y):
    xw = np.dot(x, w)
    log_term = np.log(np.exp(xw) + 1)
    print("log_term shape {}".format(log_term.shape))
    yxw_term = np.dot(y, xw)
    return np.sum(log_term - yxw_term)

def logistic_gradient(x, w, y):
    probs = logistic_function(np.dot(x, w))
    return np.dot(x.T, probs-y)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression"""
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=32):
            xw = tx_batch.dot(w)
            preds = logistic_function(xw)
            grad = logistic_gradient(tx_batch, w, y_batch)
            w -= gamma * grad
            if True:
                print("iter {} loss {}".format(n_iter, logistic_loss(tx_batch, w, y_batch)))
    return w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
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
