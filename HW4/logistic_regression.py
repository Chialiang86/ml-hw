import numpy as np
import sys
import copy
import matplotlib.pyplot as plt
from random_data_generator import univariate_gaussian_data_generator

def sigmoid(phi, w):
    return 1 / (1 + np.exp(- phi@w))

def set_cluster(N, mx, my, vx, vy, cls):
    phi = np.ones((N, 3)) # [1, x, y] = [phi_i0, phi_i1, phi_i2]
    for i in range(N):
        phi[i, 1] = univariate_gaussian_data_generator(mx, vx) # x
        phi[i, 2] = univariate_gaussian_data_generator(my, vy) # y

    y = np.zeros((N, 1)) if cls == 0 else np.ones((N, 1))

    return phi, y

def set_phi_and_y(phi0, y0, phi1, y1):

    phi = np.vstack((phi0, phi1))
    y = np.vstack((y0, y1))

    return phi, y

def gradient_descent(w, phi, y, lr=1e-2, thresh=0.05):
    grad = 1e10

    cnt = 0
    while abs(np.sqrt(np.sum(grad ** 2))) > thresh or cnt < 10000:
        cnt += 1
        gradient = phi.T @ (sigmoid(phi, w) - y)
        grad = lr * gradient
        w = w - grad 
    
    return w 

def newtons_method(w, phi, y, lr=1e-2, thresh=0.05):
    grad = 1e10

    D = np.identity(phi.shape[0])
    for i in range(phi.shape[0]):
        D[i, i] = np.exp(- phi[i] @ w) / ((1 + np.exp(- phi[i] @ w)) ** 2)
    
    hessian = (phi.T @ D) @ phi
    
    if np.linalg.cond(hessian) < 1/sys.float_info.epsilon: # check if matrix is invertable
        hessian_inv = np.linalg.inv(hessian)
    else:
        return gradient_descent(w, phi, y)

    cnt = 0
    while abs(np.sqrt(np.sum(grad ** 2))) > thresh or cnt < 10000:
        cnt += 1
        gradient = phi.T @ (sigmoid(phi, w) - y)
        grad = hessian_inv @ gradient
        w = w - grad

    return w

def predict(w, phi):
    return np.array([1 if phi_i @ w > 0 else 0 for phi_i in phi])

def confusion_matrix(y_pred, y_gt):
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    assert len(y_pred) == len(y_gt), 'predict length not equal to ground truth length!'

    for pred, gt in zip(y_pred, y_gt):
        if pred != gt:
            if pred == 1:
                FP += 1
            else :
                FN += 1
        else :
            if pred == 1:
                TP += 1
            else :
                TN += 1
    
    return np.array([[TP, FP],
                     [FN, TN]])

def print_res(title, w, conf_matrix):
    [[TP, FP], [FN, TN]] = conf_matrix
    print(title)
    print()
    print('w:')
    for _w in w:
        print(_w[0])
    print()
    print('Confusion Matrix:')
    print('             Predict cluster 1 Predict cluster 2')
    print('Is cluster 1         {:.0f}             {:.0f}'.format(conf_matrix[0, 0], conf_matrix[0, 1]))
    print('Is cluster 2         {:.0f}             {:.0f}'.format(conf_matrix[1, 0], conf_matrix[1, 1]))
    print()
    print('Sensitivity (Successfully predict cluster 1): {:.5f}'.format(TP / (TP + FN)))
    print('Specificity (Successfully predict cluster 2): {:.5f}'.format(TN / (TN + FP)))
    print()

def plot_res(phi, y_gt, y_gd, y_nm):
    plt.figure(figsize=(10, 5))

    phi_0 = phi[np.where(y_gt == 0)[0]]
    phi_1 = phi[np.where(y_gt == 1)[0]]
    plt.subplot(131)
    plt.title('Ground truth')
    plt.scatter(phi_0[:,1], phi_0[:,2], c='r')
    plt.scatter(phi_1[:,1], phi_1[:,2], c='b')

    phi_0 = phi[np.where(y_gd == 0)[0]]
    phi_1 = phi[np.where(y_gd == 1)[0]]
    plt.subplot(132)
    plt.title('Gradient descent')
    plt.scatter(phi_0[:,1], phi_0[:,2], c='r')
    plt.scatter(phi_1[:,1], phi_1[:,2], c='b')

    phi_0 = phi[np.where(y_nm == 0)[0]]
    phi_1 = phi[np.where(y_nm == 1)[0]]
    plt.subplot(133)
    plt.title('Newton\'s method')
    plt.scatter(phi_0[:,1], phi_0[:,2], c='r')
    plt.scatter(phi_1[:,1], phi_1[:,2], c='b')

    plt.savefig('logistic_refression.png')
    plt.show()


def main():
    N = int(input('number of data point: '))
    [mx1, vx1, my1, vy1] = list(map(float, input('please input mx1, vx1, my1, vy1: ').split()))
    [mx2, vx2, my2, vy2] = list(map(float, input('please input mx2, vx2, my2, vy2: ').split()))

    phi0, y0 = set_cluster(N, mx1, vx1, my1, vy1, 0)
    phi1, y1 = set_cluster(N, mx2, vx2, my2, vy2, 1)

    phi, y_gt = set_phi_and_y(phi0, y0, phi1, y1)
    w = np.random.rand(3, 1)

    w_gd = copy.copy(w)
    w_nm = copy.copy(w)
    
    w_gd_last = gradient_descent(w_gd, phi, y_gt)
    y_gd = predict(w_gd_last, phi)
    conf_matrix_gd = confusion_matrix(y_gd, y_gt)
    print_res('Gradient descent:', w_gd_last, conf_matrix_gd)

    w_nm_last = newtons_method(w_nm, phi, y_gt)
    y_nm = predict(w_nm_last, phi)
    conf_matrix_nm = confusion_matrix(y_nm, y_gt)
    print_res('Newton\'s method:', w_nm_last, conf_matrix_nm)

    plot_res(phi, y_gt, y_gd, y_nm)

if __name__=="__main__":
    main()