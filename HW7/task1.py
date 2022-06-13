import argparse
import numpy as np
from PIL import Image
import scipy.spatial.distance as dist
import cv2
import csv
import matplotlib.pyplot as plt
import os

from tomlkit import value

IMSIZE = (70, 60)

def read_data(dir : str, imsize : tuple = IMSIZE):
    paths = os.listdir(dir)
    images = []
    ys = []
    fnames = []
    for path in paths:
        img = Image.open(os.path.join(dir, path))
        img = img.resize(imsize[::-1], Image.ANTIALIAS)
        img = np.asarray(img).flatten()
        images.append(img)
        ys.append(int(path.split('.')[0][-2:]) - 1)
        fnames.append(os.path.join(dir, path))

    return np.array(images), np.array(ys), np.array(fnames)

# def pca(x, k):
    # slower implementation
    # x_mean = np.mean(x, axis=0)
    # S = (x - x_mean).T @ (x - x_mean)
    # eg_vals, eg_vecs = np.linalg.eigh(S)
    # for i in range(k):
    #     eg_vecs[:, i] = eg_vecs[:, i] / np.linalg.norm(eg_vecs[:, i])

    # ind = np.argsort(-eg_vals)[:k]
    # eg_vecs = eg_vecs[:,ind]

def pca(x, k):
    # faster implementation
    x_mean = np.mean(x, axis=0)
    S = (x - x_mean) @ (x - x_mean).T
    eg_vals, eg_vecs = np.linalg.eigh(S)
    eg_vecs = (x - x_mean).T @ eg_vecs
    for i in range(eg_vecs.shape[1]):
        eg_vecs[:, i] = eg_vecs[:, i] / np.linalg.norm(eg_vecs[:, i])
    idx = np.argsort(-eg_vals)[:k]
    eg_vecs = eg_vecs[:, idx]

    return eg_vecs, x_mean

def kernel_pca(x : np.ndarray, k : int, kernel_type : str, **kargs):

    if kernel_type == "linear":
        S = x @ x.T
    elif kernel_type == "polynomial":
        S = (kargs['gamma'] * x @ x.T + kargs['coef']) ** kargs['degree']
    elif kernel_type == "RBF":
        S = np.exp(-kargs['gamma'] * dist.cdist(x, x, 'sqeuclidean'))
    else :
        print(f'Error kernel type : {kernel_type}')
        raise Exception(f'unknown kernel type {kernel_type}')

    one_N = np.ones((x.shape[0], x.shape[0])) / x.shape[0]
    # K_C = K - I/N x K - K x I/N + I/N x K x I/n 
    S = S - one_N @ S - S @ one_N + one_N @ S @ one_N

    x_mean = np.mean(x, axis=0)
    eg_vals, eg_vecs = np.linalg.eigh(S)
    eg_vecs = (x - x_mean).T @ eg_vecs
    for i in range(eg_vecs.shape[1]):
        eg_vecs[:, i] = eg_vecs[:, i] / np.linalg.norm(eg_vecs[:, i])
    idx = np.argsort(-eg_vals)[:k]
    eg_vecs = eg_vecs[:, idx]

    return eg_vecs, x_mean

def lda(x : np.ndarray, y : np.ndarray, k : int):
    c = np.unique(y)
    x_mean = np.mean(x, axis=0)
    P_w = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    P_b = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    
    for i in c:
        x_i = x[np.where(y == i)[0], :]
        x_mean_i = np.mean(x_i, axis=0)
        P_w += (x_i - x_mean_i).T @ (x_i - x_mean_i)
        P_b += x_i.shape[0] * ((x_mean_i - x_mean).T @ (x_mean_i - x_mean))
    
    eg_vals, eg_vecs = np.linalg.eig(np.linalg.pinv(P_w) @ P_b)
    for i in range(eg_vecs.shape[1]):
        eg_vecs[:, i] = eg_vecs[:, i] / np.linalg.norm(eg_vecs[:, i])
    
    idx = np.argsort(-eg_vals)[:k]
    eg_vecs = eg_vecs[:, idx].real
    
    return eg_vecs

# gamma=0.01, coef=0, degree=3
def kernel_lda(x : np.ndarray, y : np.ndarray, k : int, kernel_type : str = None, **kargs):
    
    c = np.unique(y)
    n_features = x.shape[1]

    num_c = {i:0 for i in c}
    kernel_c = {i:np.zeros((n_features, n_features)) for i in c}
    kernel_all = np.zeros((n_features, n_features))

    for i in c:
        ind_c = (y == i).astype(int) # ex : [1, 0, 0, 1, 1, ...]
        x_c = x[ind_c]
        num_c[i] = np.sum(ind_c)
        if kernel_type == 'linear':
            kernel_c[i] = x_c.T @ x_c
        elif kernel_type == 'polynomial':
            kernel_c[i] = (kargs['gamma'] * x_c.T @ x_c + kargs['coef']) ** kargs['degree']
        elif kernel_type == 'RBF':
            kernel_c[i] = np.exp(-kargs['gamma'] * dist.cdist(x_c.T, x_c.T, 'sqeuclidean'))
        else:
            print(f'Error kernel type : {kernel_type}')
            raise Exception(f'unknown kernel type {kernel_type}')
        
    if kernel_type == 'linear':
        kernel_all = x.T @ x
    elif kernel_type == 'polynomial':
        kernel_all = (kargs['gamma'] * x_c.T @ x_c + kargs['coef']) ** kargs['degree']
    elif kernel_type == 'RBF':
        kernel_all = np.exp(-kargs['gamma'] * dist.cdist(x.T, x.T, 'sqeuclidean'))
    else:
        print(f'Error kernel type : {kernel_type}')
        raise Exception(f'unknown kernel type {kernel_type}')

    # for kernel_version N
    N = np.zeros((n_features, n_features))
    I_n = np.identity(n_features)
    one_n = np.ones((n_features, n_features))
    for i in c:
        # N = sum( K_c (I - 1/n_c) K_c.T)
        N += kernel_c[i] @ ( I_n - (1 / num_c[i]) * one_n ) @ kernel_c[i].T
    
    # for kernel_version M
    M = np.zeros((n_features, n_features))
    M_i = {i:np.zeros(n_features) for i in c}
    for i in c:
        M_i[i] = (np.sum(kernel_c[i], axis=1) / num_c[i]).reshape(n_features, 1)
    M_star = np.mean(kernel_all, axis=1).reshape(n_features, 1)
    for i in c:
        M += num_c[i] * (M_star - M_i[i]) @ (M_star - M_i[i]).T

    eg_vals, eg_vecs = np.linalg.eig(np.linalg.pinv(N) @ M)
    for i in range(eg_vecs.shape[1]):
        eg_vecs[:, i] = eg_vecs[:, i] / np.linalg.norm(eg_vecs[:, i])

    idx = np.argsort(eg_vals)[::-1]
    eg_vecs = eg_vecs[:, idx][:, :k].real

    return eg_vecs

def plot_faces(x : np.ndarray, x_mean : np.ndarray, W : np.ndarray, method : str, output_dir : str, imsize : tuple = IMSIZE):
    if x_mean is None:
        x_mean = np.zeros(x.shape[1])

    k = W.shape[1]

    assert x.shape[0] == 10, 'we just plot for 10 random images'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # for eg faces
    plt.clf()
    for i in range(k // 5):
        for j in range(5):
            idx = i * 5 + j
            plt.subplot(k // 5, 5, idx + 1)
            plt.title(f'{idx + 1}')
            plt.imshow(W[:, idx].reshape(imsize), cmap='gray')
            plt.axis('off')
    face_path = os.path.join(output_dir, f'{method}_face.png')
    plt.savefig(face_path)
    print(f'{face_path} saved')

    # for reconstruction
    reconstruction = (x - x_mean) @ W @ W.T + x_mean # nxk @ kxp = nxp

    plt.clf()
    for j in range(10):
        # for ture faces
        plt.subplot(2, 10, j + 1)
        plt.imshow(x[j].reshape(imsize), cmap='gray')
        plt.axis('off')
        # for reconstruction faces
        plt.subplot(2, 10, j + 11 )
        plt.imshow(reconstruction[j].reshape(imsize), cmap='gray')
        plt.axis('off')
    reconstruction_path = os.path.join(output_dir, f'{method}_reconstruction.png') 
    plt.savefig(reconstruction_path)
    print(f'{reconstruction_path} saved')

def face_recoginition_knn(z_train : np.ndarray, y_train : np.ndarray, z_test : np.ndarray, y_test : np.ndarray,
                         k_list : list, title : str, out_path : str):

    # computing acc by KNN algorithm
    distance = dist.cdist(z_test, z_train)
    acc = {k:0.0 for k in k_list}
    for k in k_list:

        for i in range(y_test.shape[0]):
            k_ind = np.argsort(distance[i])[:k]
            vals, counts = np.unique(y_train[k_ind], return_counts=True)
            y_pred = vals[np.argmax(counts)]
            acc[k] += float(y_pred == y_test[i])
        acc[k] /= y_test.shape[0]

    # plot the results
    fig, ax = plt.subplots()
    keys = list(acc.keys())
    values = list(acc.values())
    
    # add text
    bar = plt.bar(keys, values)
    for i, rect in enumerate(bar):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.02*h, \
            np.round(100 * values[i]) / 100, ha='center', va='bottom', rotation=0) 

    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xlabel('K')
    ax.set_ylabel('ACC')
    plt.savefig(out_path)
    print(f'{out_path} saved')


def main(args):

    try:
        task = {1:'PCA and LDA', 2:'show prediction', 3:'kernel PCA and LDA'}[args.task]
        if task == 'kernel PCA and LDA':
            kernel_type = {1:'linear', 2:'polynomial', 3:'RBF'}[args.kernel]
    except KeyError:
        print('KeyError : task number should be [1,2,3], kernel number should be [1,2,3]')

    training_dir = 'Yale_Face_Database/Training'
    testing_dir = 'Yale_Face_Database/Testing'
    x_train, y_train, fnames = read_data(training_dir, imsize=IMSIZE)
    x_test, y_test, fnames = read_data(testing_dir, imsize=IMSIZE)

    if task == 'PCA and LDA':
        randind = np.random.choice(len(fnames), 10)
        x_train_chosen = x_train[randind]
        
        eg_output_dir = 'output/task1/eigenfaces'
        print(f'computing PCA ...')
        eg_faces, x_mean = pca(x_train, 25)
        plot_faces(x_train_chosen, x_mean, eg_faces, 'pca', eg_output_dir, IMSIZE)

        fs_output_dir = 'output/task1/fisherfaces/'
        print(f'computing LDA ...')
        eg_faces50, x_mean = pca(x_train, 50) 
        x_train_reduced = (x_train - x_mean) @ eg_faces50
        fs_faces = lda(x_train_reduced, y_train, 25)
        fs_faces = eg_faces50 @ fs_faces
        plot_faces(x_train_chosen, x_mean, fs_faces, 'lda', fs_output_dir, IMSIZE)
        # fs_faces = lda(x_train, y_train, 25)
        # plot_faces(x_train_chosen, x_mean, fs_faces, 'lda_explicit', fs_output_dir, IMSIZE)

    elif task == 'show prediction':
        k_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        pca_output_file = 'output/task1/PCA_KNN.png'
        eg_faces, x_mean = pca(x_train, 25)
        z_train = (x_train - x_mean) @ eg_faces # reduction from nxp -> nxk
        z_test  = (x_test - x_mean) @ eg_faces # reduction from nxp -> nxk
        face_recoginition_knn(z_train, y_train, z_test, y_test, k_list, 'PCA prediction', pca_output_file)

        lda_output_file = 'output/task1/LDA_KNN.png'
        eg_faces50, x_mean = pca(x_train, 50) 
        x_train_reduced = (x_train - x_mean) @ eg_faces50
        fs_faces = lda(x_train_reduced, y_train, 25)
        fs_faces = eg_faces50 @ fs_faces
        z_train = x_train @ fs_faces # reduction from nxp -> nxk
        z_test  = x_test @ fs_faces # reduction from nxp -> nxk
        face_recoginition_knn(z_train, y_train, z_test, y_test, k_list, 'LDA prediction', lda_output_file)
    
    elif task == 'kernel PCA and LDA':

        randind = np.random.choice(len(fnames), 10)
        x_train_chosen = x_train[randind]
        k_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        print(f'computing kernel PCA ...')
        eg_faces, x_mean = kernel_pca(x_train, 25, kernel_type=kernel_type, gamma=0.001, coef=0, degree=3)
        z_train = (x_train - x_mean) @ eg_faces # reduction from nxp -> nxk
        z_test  = (x_test - x_mean) @ eg_faces # reduction from nxp -> nxk

        eg_output_dir = f'output/task1/{kernel_type}_eigenfaces'
        pca_output_file = f'output/task1/{kernel_type}_PCA_KNN.png'
        plot_faces(x_train_chosen, x_mean, eg_faces, 'pca', eg_output_dir, IMSIZE)
        face_recoginition_knn(z_train, y_train, z_test, y_test, k_list, 'PCA prediction', pca_output_file)

        print(f'computing kernel LDA ...')
        eg_faces50, x_mean = kernel_pca(x_train, 50, kernel_type=kernel_type, gamma=0.001, coef=0, degree=3)
        x_train_reduced = (x_train - x_mean) @ eg_faces50
        fs_faces = lda(x_train_reduced, y_train, 25)
        fs_faces = eg_faces50 @ fs_faces
        # fs_faces = kernel_lda(x_train, y_train, 25,  kernel_type=kernel_type, gamma=0.001, coef=0, degree=3)
        z_train = x_train @ fs_faces # reduction from nxp -> nxk
        z_test  = x_test @ fs_faces # reduction from nxp -> nxk

        fs_output_dir = f'output/task1/{kernel_type}_fisherfaces/'
        lda_output_file = f'output/task1/{kernel_type}_LDA_KNN.png'
        plot_faces(x_train_chosen, None, fs_faces, 'lda', fs_output_dir, IMSIZE)
        face_recoginition_knn(z_train, y_train, z_test, y_test, k_list, 'LDA prediction', lda_output_file)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', type=int, default=1)
    parser.add_argument('--kernel', '-k', type=int, default=0)
    args = parser.parse_args()
    main(args)