import numpy as np
from random_data_generator import univariate_gaussian_data_generator as ug

def update(mean, var, data_new, n):
    mean_new = mean + (data_new - mean) / n
    var_new = var + ((data_new - mean) ** 2) / n -  var / (n - 1)
    return mean_new, var_new


def main():
    [m, s] = list(map(float, input('[m, s] = ').split()))
    print(f'Data point source function: N({m}, {s})')

    val = ug(m, s)
    n = 1
    mean, var = val, 0
    mean_old, var_old = 1e10, 1e10
    print('Mean = ', mean, '   Variance = ', var)
    while (abs(mean - mean_old) > 1e-3 or abs(var - var_old) > 1e-3) :
        n += 1
        val = ug(m, s)
        mean_old, var_old = mean, var
        mean, var = update(mean_old, var_old, val, n)
        print('Add data point: ', val)
        print('Mean = ', mean, '   Variance = ', var)

if __name__=="__main__":
    main()