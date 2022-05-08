from random import uniform
import numpy as np
import matplotlib.pyplot as plt


def univariate_gaussian_data_generator(m, s):
    n = 12
    val = 0.0
    '''By the Central Limit Theorem, as n increases, the Irwin–Hall distribution more and more 
    strongly approximates a Normal distribution with mean mean = n/2, and variance = n/12'''
    for i in range(n):
        val += np.random.uniform(low=0.0, high=1.0)
    val -= 6 # N(0, 1)
    val = val * (s ** 0.5) + m # N(0, 1) -> N(m, s)

    return val 


def polynomial_basis_linear_model_data_generator(N, a, w):
    x = np.random.uniform(low=-60.0, high=60.0)
    e = univariate_gaussian_data_generator(0, a)
    wt = np.array(w)
    xm = np.array([pow(x, i) for i in range(N)]).T
    y = np.dot(wt, xm) + e # w_t * phi(x) + e
    return x, y


def main():
    mode = input(
        'a : univariate generator, b : polynomial basis linear model generator:')
    if mode == 'a':
        args = input('[Expectation Variance] = ')
        [m, s] = list(map(float, args.split(' ')))
        l = []
        for i in range(1000):
            val = univariate_gaussian_data_generator(m, s)
            print('iter {}, val = {}'.format(i + 1, val))
            l.append(val)
        l = np.array(l)
        mean = np.mean(l)
        var = np.std(l) ** 2
        print(mean, var)

    if mode == 'b':
        args = input('[N, a, w] = ')
        l = list(map(float, args.split(' ')))
        assert len(
            l) - 2 == l[0], 'N and size of w not equal : {} and {}'.format(l[0], len(l) - 2)
        N = int(l[0])
        a = l[1]
        w = l[2:]

        lx, ly = [], []
        for i in range(1000):
            x, y = polynomial_basis_linear_model_data_generator(N, a, w)
            print('iter ', i + 1, ' x = ', x, ' y = ', y)
            lx.append(x)
            ly.append(y)
        plt.scatter(lx, ly)
        plt.savefig('random_data_generator.png')
        plt.show()


if __name__ == "__main__":
    main()
