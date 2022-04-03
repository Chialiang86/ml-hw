from cv2 import mean
import copy
import matplotlib.pyplot as plt
import numpy as np
from random_data_generator import polynomial_basis_linear_model_data_generator


def phi(x, n):
    return np.array([[x ** i for i in range(n)]])

def print_mat(m):
    for r in range(m.shape[0]):
        print('    ', end='')
        for c in range(m.shape[1]):
            print(m[r][c], (',' if c < m.shape[1] - 1 else ''), end='   ')
        print()
    print()

def count_y_mean(phi_x, mean):
   return (phi_x @ mean)[0, 0]

def count_y_var(phi_x, a, cov):
   return (a + (phi_x @ cov) @ phi_x.T)[0, 0]

def plt_res(axes_id, title, x, y, a, n, mean, cov):
    x_data = np.linspace(-2, 2, 100)
    y_data_mean = [count_y_mean(phi(x, n), mean) for x in x_data]
    if title == 'Ground truth':
        y_data_pvar = [count_y_mean(phi(x, n), mean) + a for x in x_data]
        y_data_nvar = [count_y_mean(phi(x, n), mean) - a for x in x_data]
    else :
        y_data_pvar = [count_y_mean(phi(x, n), mean) + count_y_var(phi(x, n), a, cov) for x in x_data]
        y_data_nvar = [count_y_mean(phi(x, n), mean) - count_y_var(phi(x, n), a, cov) for x in x_data]

    plt.subplot(axes_id)
    plt.title(title)
    plt.xlim(-2.0, 2.0)
    plt.xticks(range(-2, 3, 1))
    plt.ylim(-20.0, 20.0)
    plt.yticks(range(-20, 30, 10))
    plt.plot(x_data, y_data_mean, 'k')
    plt.plot(x_data, y_data_pvar, 'r')
    plt.plot(x_data, y_data_nvar, 'r')
    if title != 'Ground truth':
        plt.scatter(x, y, s=5)

def main():
    args = input('[b, n, a, w] = ')
    l = list(map(float, args.split()))
    assert len(l) - 3 == l[1], 'n is not equal to len(w), n = {}, len(w) = {}'.format(n, len(w))
    b = l[0]
    n = int(l[1])
    a = l[2]
    w = l[3:]

    THRESHOLD = 1e-2
    ITER_MIN = 1e3

    lx, ly = [], []
    
    lmda = np.identity(n) / b
    cov = np.identity(n) / b
    mean = np.array([0] * n).reshape((n, 1))
    var_pred = 1e10
    cnt = 0
    while True:
        cnt += 1
        x, y = polynomial_basis_linear_model_data_generator(n, a, w)
        lx.append(x)
        ly.append(y)

        phi_x = phi(x, n)
        cov = np.linalg.inv(a * phi_x.T @ phi_x + lmda) # covariance = (a * X_t * X + lambda)^(-1)
        mean = cov @ (a * phi_x.T * y + lmda @ mean) # mean = covariance * (a * X_t * y + lambda * mean_pre)
        lmda = a * phi_x.T @ phi_x + lmda # lambda = a * X_t * X + lambda_pre

        var_prev = var_pred
        m_pred = count_y_mean(phi_x, mean)
        var_pred = count_y_var(phi_x, a, cov)
        if abs(var_pred - var_prev) < THRESHOLD and cnt > ITER_MIN:
            break
        
        print('------------------------------------------')
        print(f'Add data point ({x:>.5f}, {y:>.5f}):\n')
        print('Posterior mean:')
        print_mat(mean)
        print('Posterior variance:')
        print_mat(cov)
        print(f'Predictive distribution ~ N({m_pred:>.5f}, {var_pred:>.5f})\n')

        if cnt == 10:
            lx_10 = copy.copy(lx) 
            ly_10 = copy.copy(ly) 
            cov_10 = copy.copy(cov)
            mean_10 = copy.copy(mean)
        elif cnt == 50:
            lx_50 = copy.copy(lx) 
            ly_50 = copy.copy(ly) 
            cov_50 = copy.copy(cov)
            mean_50 = copy.copy(mean)


    w = np.array(w).reshape((n, 1))

    plt.figure(figsize=(10, 10))
    plt_res(221, 'Ground truth', None, None, a, n, w, None)
    plt_res(222, 'Predict result', lx, ly, a, n, mean, cov)
    plt_res(223, 'After 10 incomes', lx_10, ly_10, a, n, mean_10, cov_10)
    plt_res(224, 'After 50 incomes', lx_50, ly_50, a, n, mean_50, cov_50)
    plt.savefig('baysian_linear_regression.png')
    plt.show()


if __name__=='__main__':
    main()