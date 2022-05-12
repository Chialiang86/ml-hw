from cProfile import label
from turtle import distance
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import scipy.spatial.distance as dist
from libsvm.svmutil import *

def task1(X_train, Y_train, X_test, Y_test):
    s_dict = {'C-SVC': 0, 'nu-SVC':1, 'one-class SVM':2, 'epsilon-SVR':3, 'nu-SVR':4}
    t_dict = {'linear': 0, 'polynomial':1, 'RBF':2, 'sigmoid':3}
    
    f_out = open('task2.1_out.txt', 'w')    
    
    c_default = 1
    c_list = [c_default / 100000, c_default / 10000, c_default / 1000, c_default / 100, c_default / 10, c_default,
               c_default * 10, c_default * 100, c_default * 1000, c_default * 10000, c_default / 100000]
    acc_map = np.zeros((3, len(c_list)))

    for i, c in enumerate(c_list):
        # -s 0 (C-SVC) -t 0 (linear) -c 10 (num classes) 
        kernel_type = 'linear'
        param = '-s {} -t {} -c {}'.format(s_dict['C-SVC'], t_dict[kernel_type], c)
        m = svm_train(Y_train, X_train, param)
        p_label, p_acc, p_val = svm_predict(Y_test, X_test, m)
        ACC, MSE, SCC = evaluations(Y_test, p_label)
        acc_map[0, i] = ACC
        f_out.write('[kernel type = {}, c = {}, ACC={}, MSE={}, SCC={}]\n'.format(kernel_type, c, ACC, MSE, SCC))
        
        # -s 0 (C-SVC) -t 1 (polynomial) -c 10 (num classes) 
        kernel_type = 'polynomial'
        param = '-s {} -t {} -c {}'.format(s_dict['C-SVC'], t_dict[kernel_type], c)
        m = svm_train(Y_train, X_train, param)
        p_label, p_acc, p_val = svm_predict(Y_test, X_test, m)
        ACC, MSE, SCC = evaluations(Y_test, p_label)
        f_out.write('[kernel type = {}, c = {}, ACC={}, MSE={}, SCC={}]\n'.format(kernel_type, c, ACC, MSE, SCC))
        acc_map[1, i] = ACC
        
        # -s 0 (C-SVC) -t 2 (RBF) -c 10 (num classes) 
        kernel_type = 'RBF'
        param = '-s {} -t {} -c {}'.format(s_dict['C-SVC'], t_dict[kernel_type], c)
        m = svm_train(Y_train, X_train, param)
        p_label, p_acc, p_val = svm_predict(Y_test, X_test, m)
        ACC, MSE, SCC = evaluations(Y_test, p_label)
        f_out.write('[kernel type = {}, c = {}, ACC={}, MSE={}, SCC={}]\n'.format(kernel_type, c, ACC, MSE, SCC))
        acc_map[2, i] = ACC
    
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(15)

    x_axis = np.arange(acc_map.shape[1])
    plt.title('Task2.1 result (accuracy in %)')
    plt.xticks(x_axis, c_list)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    bar_linear          = plt.bar(x_axis - 0.25, acc_map[0], 0.25, label = 'linear')
    bar_RBF             = plt.bar(x_axis - 0., acc_map[1], 0.25, label = 'polynomial')
    bar_linear_plus_RBF = plt.bar(x_axis + 0.25, acc_map[2], 0.25, label = 'RBF')
    for rect in bar_linear + bar_RBF + bar_linear_plus_RBF:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.1f}', ha='center', va='bottom', fontsize='x-small')

    plt.xlabel('c')
    plt.ylim(0, 120)
    plt.ylabel('acc')
    plt.legend()

    plt.savefig('task2.1_three_kernel.png')
    plt.show()

def plot_res(data, title, x_label, y_label, z_label, c):
    if len(data.shape) == 3:
        for i in range(data.shape[0]):
            fig, ax = plt.subplots(figsize=(10, 10))
            plt.title(f'{title} : degree = {x_label[i]}, c = {c} (accuracy in %)')
            plt.imshow(data[i].T)
            plt.xlabel('gamma')
            plt.ylabel('coef')
            plt.xticks(np.arange(len(y_label)), labels=y_label)
            plt.yticks(np.arange(len(z_label)), labels=z_label)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            for j in range(len(y_label)):
                for k in range(len(z_label)):
                    text = plt.text(j, k, f'{data[i, j, k]}%', ha='center', va='center', color='w')
            plt.savefig(f'task2.2_{title}_{c}_{x_label[i]}.png')
        
    elif len(data.shape) == 2:
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.title(f'{title} (accuracy in %)')
        plt.imshow(data.T)
        plt.xlabel('gamma')
        plt.ylabel('c')
        plt.xticks(np.arange(len(x_label)), labels=x_label)
        plt.yticks(np.arange(len(y_label)), labels=y_label)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(len(x_label)):
            for j in range(len(y_label)):
                text = plt.text(i, j, f'{data[i, j]}%', ha='center', va='center', color='w')
        
        plt.savefig(f'task2.2_{title}.png')

def task2(X_train, Y_train, X_test, Y_test):
    # for polynomial (gamma*u'*v + coef0)^degree
    gamma_default = 0.00125
    coef_default = 0
    degree_default = 3
    c_default = 1

    f_out = open('task2.2_out.txt', 'w')

    degree_list = [degree_default - 1, degree_default, degree_default + 1]
    gamma_list = [gamma_default / 100000, gamma_default / 10000, gamma_default / 1000, gamma_default / 100, gamma_default / 10, gamma_default, 
                  gamma_default * 10, gamma_default * 100, gamma_default * 1000, gamma_default * 10000, gamma_default * 100000]
    coef_list = [coef_default - 1000, coef_default - 100, coef_default - 10, coef_default - 1, coef_default - 0.1, coef_default,
                 coef_default + 0.1, coef_default + 1, coef_default + 10, coef_default + 100, coef_default + 1000]

    polynomial_acc_map = np.zeros((len(degree_list), len(gamma_list), len(coef_list)))
    c = 1
    # -s 0 (C-SVC) -t 1 (polynomial) -c 10 (num classes) 
    for i, degree in enumerate(degree_list):
        for j, gamma in enumerate(gamma_list):
            for k, coef in enumerate(coef_list):
                print(f'[degree = {degree}, coef = {coef},gamma = {gamma}]\n')
                param = '-s 0 -t 1 -c {} -g {} -r {} -d {}'.format(c, gamma, coef, degree)
                m = svm_train(Y_train, X_train, param)
                p_label, p_acc, p_val = svm_predict(Y_test, X_test, m)
                ACC, MSE, SCC = evaluations(Y_test, p_label)
                f_out.write(f'[param : {param}, ACC = {ACC}, MSE = {MSE}, SCC = {SCC}]\n')
                polynomial_acc_map[i, j, k] = np.round(100 * ACC) / 100
    
    f_out.write('\nPolynomial Based Kernel Analysis\n')
    for i in range(polynomial_acc_map.shape[0]):
        f_out.write(f'\ndegree = {degree_list[i]}\n')
        for j in range(polynomial_acc_map.shape[1]):
            for k in range(polynomial_acc_map.shape[2]):
                f_out.write(f'{polynomial_acc_map[i, j, k]} ')
            f_out.write('\n')
    f_out.write('\n')
    plot_res(polynomial_acc_map, 'polynomial', degree_list, gamma_list, coef_list, c)

    # for RBF exp(-gamma*|u-v|^2)
    gamma_list = [gamma_default / 100000, gamma_default / 10000, gamma_default / 1000, gamma_default / 100, gamma_default / 10, gamma_default, 
                  gamma_default * 10, gamma_default * 100, gamma_default * 1000, gamma_default * 10000, gamma_default * 100000]
    c_list = [c_default / 100000, c_default / 10000, c_default / 1000, c_default / 100, c_default / 10, c_default,
               c_default * 10, c_default * 100, c_default * 1000, c_default * 10000, c_default / 100000]
    RBF_acc_map = np.zeros((len(gamma_list), len(c_list)))

    # -s 0 (C-SVC) -t 2 (RBF) -c 10 (num classes) 
    for i, gamma in enumerate(gamma_list):
        for j, c in enumerate(c_list):
            print(f'[gamma = {gamma}, c = {c}]\n')
            param = '-s 0 -t 2 -c {} -g {}'.format(c, gamma)
            m = svm_train(Y_train, X_train, param)
            p_label, p_acc, p_val = svm_predict(Y_test, X_test, m)
            ACC, MSE, SCC = evaluations(Y_test, p_label)
            f_out.write(f'[param : {param}, ACC = {ACC}, MSE = {MSE}, SCC = {SCC}]\n')
            RBF_acc_map[i, j] = np.round(100 * ACC) / 100

    f_out.write('\nRBF Based Kernel Analysis\n')
    for i in range(RBF_acc_map.shape[0]):
        for j in range(RBF_acc_map.shape[1]):
            f_out.write(f'{RBF_acc_map[i, j]} ')
        f_out.write('\n')
    f_out.write('\n')
    
    # plot result to heatmap
    plot_res(RBF_acc_map, 'RBF', gamma_list, c_list, None, c)

    print('process completed.')

def kernel_linear(x, y, gamma):
    return x @ y.T

def kernel_RBF(x, y, gamma):
    return np.exp(-gamma * dist.cdist(x, y))

def kernel_linear_plus_RBF(x, y, gamma):
    return kernel_linear(x, y, gamma) + kernel_RBF(x, y, gamma)

def kernelize(X_train, X_test, func, gamma):
    '''
    link : https://github.com/cjlin1/libsvm, keyword : precomputed kernel
    Assume the original training data has three four-feature
	instances and testing data has one instance:

	15  1:1 2:1 3:1 4:1
	45      2:3     4:3
	25          3:1

	15  1:1     3:1

	If the linear kernel is used, we have the following new
	training/testing sets:8

	15  0:1 1:4 2:6  3:1
	45  0:2 1:6 2:18 3:0
	25  0:3 1:1 2:0  3:1

	15  0:? 1:2 2:0  3:1

	? can be any value.
    '''
    
    print('computing kernel for training')
    training_kernel = func(X_train, X_train, gamma)
    testing_kernel = func(X_test, X_train, gamma)
    
    training_kernelized = []
    for i in range(training_kernel.shape[0]):
        training_dict = {}
        training_dict[0] = i+1
        for j in range(training_kernel.shape[1]):
            training_dict[j+1] = training_kernel[i, j]
        training_kernelized.append(training_dict)
    
    testing_kernelized = []
    for i in range(testing_kernel.shape[0]):
        testing_dict = {}
        testing_dict[0] = i+1
        for j in range(testing_kernel.shape[1]):
            testing_dict[j+1] = testing_kernel[i, j]
        testing_kernelized.append(testing_dict)
    
    return training_kernelized, testing_kernelized

def task3(X_train, Y_train, X_test, Y_test):

    gamma_default = 0.00125
    gamma_list = [gamma_default / 100000, gamma_default / 10000, gamma_default / 1000, gamma_default / 100, gamma_default / 10, gamma_default, 
                  gamma_default * 10, gamma_default * 100, gamma_default * 1000, gamma_default * 10000, gamma_default * 100000]
    func_list = [kernel_linear, kernel_RBF, kernel_linear_plus_RBF]
    acc_map = np.zeros((len(func_list), len(gamma_list)))

    # precompute kernel

    f_out = open('task2.3_out.txt', 'w')

    # run for all gamma
    for i, func in enumerate(func_list):
        for j, gamma in enumerate(gamma_list):
            X_train_kenelized, X_test_kernelized = kernelize(X_train, X_test, func, gamma)
            param = '-s 0 -t 4 -c 1 -g {}'.format(gamma)
            m = svm_train(Y_train, X_train_kenelized, param)
            p_label, p_acc, p_val = svm_predict(Y_test, X_test_kernelized, m)
            ACC, MSE, SCC = evaluations(Y_test, p_label)
            f_out.write(f'[kernel : {func}, gamma : {gamma}, ACC = {ACC}, MSE = {MSE}, SCC = {SCC}]\n')
            acc_map[i, j] = np.round(100 * ACC) / 100

    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(15)

    x_axis = np.arange(acc_map.shape[1])
    plt.title('Task2.2 result (accuracy in %)')
    plt.xticks(x_axis, gamma_list)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    bar_linear          = plt.bar(x_axis - 0.25, acc_map[0], 0.25, label = 'linear')
    bar_RBF             = plt.bar(x_axis - 0., acc_map[1], 0.25, label = 'RBF')
    bar_linear_plus_RBF = plt.bar(x_axis + 0.25, acc_map[2], 0.25, label = 'linear+RBF')
    for rect in bar_linear + bar_RBF + bar_linear_plus_RBF:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.1f}', ha='center', va='bottom', fontsize='x-small')

    plt.xlabel('gamma')
    plt.ylim(0, 120)
    plt.ylabel('acc')
    plt.legend()

    plt.savefig('task2.3_linear+RBF.png')
    plt.show()

    
def main():

    # open dataset
    fX_train = open('data/X_train.csv', 'r')
    fY_train = open('data/Y_train.csv', 'r')
    fX_test = open('data/X_test.csv', 'r')
    fY_test = open('data/Y_test.csv', 'r')

    # config dataset
    X_train = np.array([[float(val) for val in line.split(',')] for line in fX_train])
    Y_train = np.array([float(val) for val in fY_train])
    X_test = np.array([[float(val) for val in line.split(',')] for line in fX_test])
    Y_test = np.array([float(val) for val in fY_test])

    option = int(input('please input task id (1 or 2 or 3):'))
    
    if option == 1:
        # for question 2-1
        task1(X_train, Y_train, X_test, Y_test)
    elif option == 2:
        # for question 2-2
        task2(X_train, Y_train, X_test, Y_test)
    elif option == 3:
        # for question 2-3
        task3(X_train, Y_train, X_test, Y_test)

if __name__=="__main__":
    main()