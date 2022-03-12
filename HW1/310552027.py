import numpy as np
import random
import argparse
import matplotlib.pyplot as plt

# util functions for matrix operation
#######################################################################################

# transpose matrix
def tran_mat(A):
    assert A is not None, 'the matrix is empty'

    ret = np.zeros((A.shape[1], A.shape[0]))
    for i in range(A.shape[0]):
        ret[:, i] = A[i, :]
    return ret 

# matrix multiplication
def mul_mat(A, B):
    assert A.shape[1] == B.shape[0], 'matrix A\'s column num not equal to B\'s row num'

    A_row = A.shape[0]
    B_col = B.shape[1]
    
    ret = np.zeros((A_row, B_col))
    for r in range(A_row):
        for c in range(B_col):
            ret[r, c] = sum(A[r, :] * B[:, c])
    return ret 

# matrix invertion
def inv_mat(A):
    assert A.shape[0] == A.shape[1], 'A is not a square matrix'

    n = A.shape[0]
    A_aug = np.hstack((A, np.identity(n)))
    for di in range(n):
        pivot_row = di
        non_zero_row = -1

        # find non zero row in the di'th column
        for row in range(di, A_aug.shape[0]):
            if A_aug[row, di] != 0:
                non_zero_row = row
                break
        
        # no need to eliminate
        if non_zero_row == -1:
            # no solution
            return []
        
        # setting pivot
        swap_row(A_aug, pivot_row, non_zero_row)

        for row in range(0, A_aug.shape[0]):
            if A_aug[row, di] != 0 and row != pivot_row:
                scale = -A_aug[row, di] / A_aug[pivot_row, pivot_row]
                add_to_row(A_aug, pivot_row, row, scale)
        
        scale_row(A_aug, pivot_row, 1.0 / A_aug[pivot_row, pivot_row]) 

    A_inv = A_aug[:,n:]

    return A_inv

# elementary matrix : multiply one row
def scale_row(A, i, scale): # A[i] *= scale
    assert i < A.shape[0] and i >=0 , 'i or j is illegal'
    A[i] *= scale

# elementary matrix : swap two rows
def swap_row(A, i, j): # A[i] <-> A[j]
    assert i < A.shape[0] and i >=0 and j < A.shape[0] and j >=0, 'i or j is illegal'
    if i == j:
        return 

    for ele in range(A.shape[1]):
        tmp = A[i, ele]
        A[i, ele] = A[j, ele]
        A[j, ele] = tmp 

# elementary matrix : add scaled one row to another row 
def add_to_row(A, i, j, scale): # A[j] += A[i] * scale
    assert i < A.shape[0] and i >=0 and j < A.shape[0] and j >=0, 'i or j is illegal'
    addrr = scale * A[i]
    A[j] += addrr

#######################################################################################

# A = [X^0 X^1 X^2 ... X^n] -> shape = (N, n)
# b = [y_0 y_1 y_2 ... y_N].T -> shape = (N, 1)
def buildAb(x, y, n):
    xnum = len(x)

    A = np.array([pow(x[0], power) for power in range(n)])
    b = np.array([y[0]])

    for i in range(1, xnum):
        Arow_i = np.array([pow(x[i], power) for power in range(n)])
        brow_i = np.array(y[i])

        A = np.vstack((A, Arow_i))
        b = np.vstack((b, brow_i))
    
    return A, b
    
def solve_rLSE(A, b, lmda):
    ATA = mul_mat(tran_mat(A), A)
    ATA_lmdaI = ATA + lmda * np.identity(ATA.shape[0])
    ATb = mul_mat(tran_mat(A), b)
    ret = mul_mat(inv_mat(ATA_lmdaI), ATb)
    return ret

def solve_Newton(A, b, init):
    solution_new = init 
    square_error = 1.0
    while (square_error > 1e-10) :
        solution_tmp = solution_new 
        gradient = 2 * mul_mat(mul_mat(tran_mat(A), A), solution_tmp)  - 2 * mul_mat(tran_mat(A), b)
        hessian = 2 * mul_mat(tran_mat(A), A) 
        solution_new = solution_tmp - mul_mat(inv_mat(hessian), gradient)
        square_error = sum((solution_new - solution_tmp) ** 2)
    return solution_new

def format_solution(a):
    sol_str = ""
    for i in range(len(a) - 1, 0, -1):
        sol_str += "{}X^{} + ".format(a[i, 0], i)
    sol_str += "{}".format(a[0, 0])
    return sol_str

def format_round_solution(a):
    sol_str = ""
    for i in range(len(a) - 1, 0, -1):
        sol_str += "{:.2f}X^{} + ".format(a[i, 0], i)
    sol_str += "{:.2f}".format(a[0, 0])
    return sol_str

def count_square_error(a, x, y):
    ret = 0.0
    num_data = len(x)
    for i in range(num_data):
        y_predict = 0
        for p in range(len(a)):
            y_predict += a[p] * pow(x[i], p)
        ret += pow(np.abs(y_predict - y[i]), 2)
        
    return ret[0]

def gen_curve_points(x, a):
    x_min, x_max = min(x) - 0.1 * (max(x) - min(x)), max(x) + 0.1 * (max(x) - min(x))
    x_data = x_min + (x_max - x_min) / 10000 * np.array(range(10000))
    y_data = []
    for x_tmp in x_data:
        y_tmp = 0
        for pow, coef in enumerate(a):
            y_tmp += coef * x_tmp ** pow
        y_data.append(y_tmp)
    return np.array(x_data), np.array(y_data)

def visualize_result(x, y, a_LSE, err_LSE, a_Newton, err_Newton):
    _, axes = plt.subplots(2, 1, figsize=(8, 10))
    x_LSE, y_LSE = gen_curve_points(x, a_LSE)
    x_Newton, y_Newton = gen_curve_points(x, a_Newton)
    axes[0].set_title('LSE method (error : {:.5f})'.format(err_LSE))
    axes[0].scatter(x, y, label='data point')
    axes[0].plot(x_LSE, y_LSE, 'k', label=format_round_solution(a_LSE))
    axes[0].grid()
    axes[0].legend()
    axes[1].set_title('Newton\'s method (error : {:.5f})'.format(err_Newton))
    axes[1].scatter(x, y, label='data point')
    axes[1].plot(x_Newton, y_Newton, 'k', label=format_round_solution(a_Newton))
    axes[1].legend()
    axes[1].grid()

    plt.savefig('result.png')
    plt.show()

def main(args):
    n = args.n
    lmda = args.l

    f_in = open(args.file)
    lines = f_in.readlines()

    x = []
    y = []
    for line in lines:
        x.append(float(line.split(',')[0]))
        y.append(float(line.split(',')[1]))
    
    x = np.array(x)
    y = np.array(y)

    A, b = buildAb(x, y, n)

    a_LSE = solve_rLSE(A, b, lmda)
    solution_LSE = format_solution(a_LSE)
    err_LSE = count_square_error(a_LSE, x, y)

    print('LSE:')
    print('Fitting line:', solution_LSE)
    print('Total error: {:.10f}'.format(err_LSE))
    
    init = np.reshape(np.array([random.random() for i in range(n)]), (n , 1))
    a_Newton = solve_Newton(A, b, init)
    solution_Newton = format_solution(a_Newton)
    err_Newton = count_square_error(a_Newton, x, y)

    print('\nNewton\'s Method:')
    print('Fitting line:', solution_Newton)
    print('Total error: {:.10f}'.format(err_Newton))
    
    visualize_result(x, y, a_LSE, err_LSE, a_Newton, err_Newton)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', default='testfile.txt', type=str, help='the input file of the program')
    parser.add_argument('n', type=int, help='the integer number of n')
    parser.add_argument('l', type=int, help='the float number of lambda')
    args = parser.parse_args()

    main(args)
