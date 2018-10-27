import numpy as np

# We use globals to make iterative functions display
# things like validation accuracy conditionally (we
# must use globals since interface of the function is
# fixed in project description PDF)
X_VALID = None
Y_VALID = None

def set_validation_dataset(x, y):
    global X_VALID
    global Y_VALID
    X_VALID = x
    Y_VALID = y

VERBOSE_OUTPUT = True

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_mse_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)

def mse_gradient(y, tx, w):
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm using mse."""
    w = initial_w
    for n_iter in range(max_iters):
        grad, err = compute_gradient(y, tx, w)
        curr_mse_loss = calculate_mse(err)
        w -= gamma * grad
        if VERBOSE_OUTPUT:
            print("Gradient Descent({bi}/{ti}): gamma={g} mse-loss={l} ".format(
                bi=n_iter, ti=max_iters - 1, l=curr_mse_loss, w=w, g=gamma))
    return w, curr_mse_loss

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

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            grad, _ = compute_gradient(y_batch, tx_batch, w)
            w -= gamma * grad
            loss = compute_loss(y, tx, w)
        if VERBOSE_OUTPUT:
            print("SGD({bi}/{ti}): mse-loss={l}, gamma={gamma}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, gamma=gamma))
    return w, loss

def least_squares(y, tx):
    A = np.transpose(tx).dot(tx)
    b = tx.T.dot(y)
    return p.linalg.solve(A, b)


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
        w -= gamma * grad
        if n_iter % 100 == 0 and VERBOSE_OUTPUT:
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
