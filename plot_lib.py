# defines initial w of size of features
# w_initial = np.zeros((X_train.shape[1]), dtype=int)

import dataset as dataset
import implementations
import numpy as np
import matplotlib.pyplot as plt

def plot_train_test(train_errors, test_errors, test_range, method_name, testparam):
    plt.figure(figsize=(12,10))
    plt.semilogx(test_range, train_errors, color='b', marker='*', label="Train error")
    plt.semilogx(test_range, test_errors, color='r', marker='*', label="Test error")
    plt.xlabel(testparam,  fontsize=18)
    plt.ylabel("root mean squared error",  fontsize=18)
    plt.title(""+method_name+" - "+testparam+" testing", fontsize=14)
    leg = plt.legend(loc=1, shadow=True, prop={'size': 18})
    leg.draw_frame(False)
    plt.savefig(method_name+"_"+testparam+"_testing")

# method_name   = algorithm
# params        = specific parameters for method
# test_range    = range from to of to_test_param
# to_test_param = the parameter we want to test in the range of test_range
def hyperparameter_testing(y, x, ratio, seed, test_range, method_name, params, to_test_param):
    # define parameter
    test_range = test_range
    
    # split data into test and train
    x_tr, x_te, y_tr, y_te = dataset.split_data(x, y, ratio, seed)
    c = 0
    
    # store train and test err
    err_train = []
    err_test = []
    
    training_tr = implementations.Training(method_name, params)
    training_te = implementations.Training(method_name, params)
    
    for ind, new_param in enumerate(test_range):
        training_tr.params[to_test_param] = new_param
        w_tr, mse_tr = training_tr.fit(x_tr, y_tr)
        
        training_te.params[to_test_param] = new_param  
        w_te, mse_te = training_te.fit(x_te, y_te)
                
        err_train.append(mse_tr)
        err_test.append(mse_te)
        c += 1
        print(c, "proportion={p}, gamma={g:.3f}, Training RMSE={tr:.5f}, Testing RMSE={te:.5f}".format(
               p=ratio, g=new_param, tr=err_train[ind], te=err_test[ind]))
    plot_train_test(err_train, err_test, test_range, method_name, to_test_param)