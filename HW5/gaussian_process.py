import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from random_data_generator import polynomial_basis_linear_model_data_generator

# hyperparameter
VAR = 1
L = 1
ALPHA = 1

def quadratic_kernel(x_n, x_m, kernel_args):
    (var, l, alpha) = kernel_args 

    # rational quadratic kernel
    return var * (1 + ((x_n - x_m)**2 / (2 * alpha * l**2))) ** (-alpha)

def quadratic_kernel_mat(X, kernel_args):
    (var, l, alpha) = kernel_args 
    X_repeat_h = np.repeat(X, X.shape[0], axis=1) # (n, 1) ->  (n, n)
    X_repeat_v = np.repeat(X.T, X.shape[0], axis=0) # (1, n) -> (n, n)

    # rational quadratic kernel matrix
    return var * (1 + ((X_repeat_h - X_repeat_v)**2 / (2 * alpha * l**2))) ** (-alpha)

def gaussian_process_regression(x_samples, X, Y, beta, kernel_args):
    # get covariance matrix
    C = quadratic_kernel_mat(X, kernel_args) + (1/beta) * np.identity(X.shape[0]) # c(n, m) = k(x_n, x_m) + 1/beta * delta(n, m)
    assert np.linalg.det(C) != 0, 'C is not invertible'

    Y_up_95,Y_mean, Y_down_95 = [], [], []
    for x_sample in x_samples:

        k_x_xstar = np.array([quadratic_kernel(x_sample, x, kernel_args) for x in X]) # k(x, x*)
        k_star = quadratic_kernel(x_sample, x_sample, kernel_args) + 1/beta # k* = k(x*, x*) + 1/beta

        # count mu(x_star), std(x_star)
        mean = ((k_x_xstar.T @ np.linalg.inv(C)) @ Y)[0,0]
        std = (k_star - (k_x_xstar.T @ np.linalg.inv(C)) @ k_x_xstar)[0,0]

        Y_up_95.append(mean + 1.96 * (std**0.5))
        Y_mean.append(mean)
        Y_down_95.append(mean - 1.96 * (std**0.5))
    
    return Y_up_95, Y_mean, Y_down_95

def negative_marginal_log_likelihood(parameters, *args) :
    # args = (sign, X, Y, beta)
    [var, l, alpha] = parameters # parameters = [var, l, alpha]
    (sign, X, Y, beta) = args
    n = X.shape[0]

    # quadratic kernel matrix
    X_repeat_h = np.repeat(X, n, axis=1) # (n, 1) ->  (n, n)
    X_repeat_v = np.repeat(X.T, n, axis=0) # (1, n) -> (n, n)
    K = var * (1 + ((X_repeat_h - X_repeat_v)**2 / (2 * alpha * (l**2)))) ** (-alpha)
    
    # covariance
    C = K + (1/beta) * np.identity(n) # c(n, m) = k(x_n, x_m) + 1/beta * delta(n, m)
    C_inv = np.linalg.inv(C)

    # marginal log likelihood
    ret = -(Y.T @ C_inv) @ Y -1/2 * np.log(np.linalg.det(C)) -n/2 * np.log(2 * np.pi)
    return sign * ret[0,0]

def task1(x_samples, X, Y, beta):
    # hyper parameters
    kernel_args=(VAR, L, ALPHA)
    
    Y_up_95, Y_mean, Y_down_95 = gaussian_process_regression(x_samples, X, Y, beta, kernel_args)
    before_optimization = negative_marginal_log_likelihood([VAR, L, ALPHA], 1, X, Y, beta)
    print('before optimization : [var, l, alpha] = [{:.5f}, {:.5f}, {:.5f}], log likelihood = {}'.format(VAR, L, ALPHA, before_optimization))

    plt.scatter(X, Y, c='b', s=10)
    plt.plot(x_samples, Y_up_95, c='r')
    plt.plot(x_samples, Y_mean, c='g')
    plt.plot(x_samples, Y_down_95, c='r')
    plt.savefig('before_optimization.png')
    plt.show()


def task2(x_samples, X, Y, beta):

    # run optimization by initial guess given
    result = optimize.minimize(negative_marginal_log_likelihood, [VAR, L, ALPHA], args=(-1, X, Y, beta))
    
    [var, l, alpha] = result.x
    kernel_args=(var, l, alpha)
    Y_up_95, Y_mean, Y_down_95 = gaussian_process_regression(x_samples, X, Y, beta, kernel_args)
    after_optimization = negative_marginal_log_likelihood([var, l, alpha], 1, X, Y, beta)
    print('after optimization  : [var, l, alpha] = [{:.5f}, {:.5f}, {:.5f}], log likelihood = {}'.format(var, l, alpha, after_optimization))

    plt.scatter(X, Y, c='b', s=10)
    plt.plot(x_samples, Y_up_95, c='r')
    plt.plot(x_samples, Y_mean, c='g')
    plt.plot(x_samples, Y_down_95, c='r')
    plt.savefig('after_optimization.png')
    plt.show()

def main():
    # read raw data
    input_data = open('data/input.data', 'r')
    lines = input_data.readlines()
    data_list = np.array([[float(data.split(' ')[0]), float(data.split(' ')[1])] for data in lines])

    # config arguments
    x_samples = np.linspace(-60, 60, 1000)
    X = data_list[:, 0].reshape((data_list.shape[0], 1))
    Y = data_list[:, 1].reshape((data_list.shape[0], 1))
    beta = 5

    option = int(input('please input task id (1 or 2):'))

    if option == 1:
        # code for task 1.1
        task1(x_samples, X, Y, beta)

    elif option == 2:
        # code for task 1.2
        task2(x_samples, X, Y, beta)


if __name__=="__main__":
    main()