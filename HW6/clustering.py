import numpy as np
import cv2
import copy
import imageio
import os
import glob
from scipy.spatial.distance import cdist
import argparse

def get_flattened_imloc(shape):
    assert len(shape) > 2, f'invalid shape length {shape}'

    (xr, xc) = np.meshgrid(range(shape[0]), range(shape[1]), indexing='ij')
    xr = xr.flatten()
    xc = xc.flatten()
    return np.vstack((xr, xc)).T

def get_flattened_imrbg(img):
    assert len(img.shape) == 3, f'invalid shape length {img.shape}'
    
    return np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))

def get_gram_matrix(img_c, img_s, gamma_s, gamma_c):

    c_x = cdist(img_c, img_c) ** 2
    s_x = cdist(img_s, img_s) ** 2

    # gram matrix
    k = np.exp(-gamma_s * s_x) + np.exp(-gamma_c * c_x)

    return k 

def get_palette(k):
    dist_thresh = 100

    palette = np.zeros((k, 3))
    for j in range(k):
        repeat = True
        while repeat:
            palette[j] = np.array([np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)])
            repeat = False
            for jj in range(j):
                if np.sum(np.abs(palette[j] - palette[jj])) < dist_thresh:
                    repeat = True
                    break

    return palette

def visualize_clusters(save_path, clusters, shape, palette):
    assert len(shape) == 3, f'invalid shape size: {shape}'

    gif_frames = []
    for cluster in clusters:

        cluster_map = ((np.reshape(cluster, shape[:2]) * 50) % 256).astype(np.uint8)

        gif_frame = cv2.applyColorMap(cluster_map, cv2.COLORMAP_HSV)
        gif_frames.append(gif_frame)

    imageio.mimsave(save_path, gif_frames, fps=2)

def kmeans(data, k):
    data_size = data.shape[0]
    data_dim = data.shape[1]
    cluster_old = np.array([np.random.randint(0, k) for i in range(data_size)])
    dist_map = np.zeros((data_size, k))
    means = np.zeros((k, data_dim))
    for j in range(k):
        cond = np.where(cluster_old == j)
        means[j] = np.mean(data[cond], axis=0)
    
    cluster_frames = []
    
    diff = np.inf
    iteration = 0
    max_iteration = 1000
    thresh = 10
    while diff > thresh and iteration < max_iteration:
        iteration += 1
        
        # E step
        for j in range(k):
            dist_map[:, j] = np.sum((data - means[j]) ** 2, axis=1)
        cluster = np.argmin(dist_map, axis=1)

        # M step
        for j in range(k):
            cond = np.where(cluster == j)
            means[j] = np.mean(data[cond], axis=0)

        diff = np.sum(np.abs(cluster - cluster_old))
        cluster_old = cluster
        print(f'[kmeans iteration : {iteration}, diff : {diff}]')

        cluster_frames.append(cluster)
    
    return cluster_frames


def kernel_kmeans(gram_mat, k):
    
    data_size = gram_mat.shape[0]
    # get the diagonal value of the gram matrix to form a new list 
    gram_D = np.array([gram_mat[i, i] for i in range(data_size)])

    # initialize cluster
    cluster_old = np.array([np.random.randint(0, k) for i in range(data_size)])
    dist_map = np.zeros((data_size, k))
    indicator_map = np.zeros((data_size, k)) # orthogonal to each other
    for j in range(k):
        cond = np.where(cluster_old == j)
        indicator_map[cond, j] = 1
    
    cluster_frames = []
    diff = np.inf
    iteration = 0
    thresh = 10
    while diff > thresh:
        iteration += 1

        for j in range(k):
            Cj_num = np.sum(indicator_map[:,j]) # |Ck|
            # k(x_j, x_j)
            first_term = gram_D 
            # 2 / |Ck| * sum_n(alpha_kn * k(x_j, x_n))
            second_term = 2 / Cj_num * np.dot(gram_mat, indicator_map[:, j])
            # 1 / |Ck|^2 * sum_p(sum_q(alpha_kp * alpha_kq * k(x_p, x_q)))
            third_term = (1 / Cj_num**2) * np.dot(np.dot(indicator_map[:, j].T, gram_mat), indicator_map[:, j])
            
            dist_map[:, j] = first_term - second_term + third_term

        # assign to new cluster/indicator mat
        cluster = np.argmin(dist_map, axis=1)
        indicator_map = np.zeros((data_size, k))
        for j in range(k):
            cond = np.where(cluster == j)
            indicator_map[cond, j] = 1

        # count difference between current cluster and new cluster
        diff = np.sum(np.abs(cluster - cluster_old)) 
        cluster_old = copy.deepcopy(cluster)
        print(f'[kernel kmeans iteration : {iteration}, diff : {diff}]')

        cluster_frames.append(cluster)
    
    return cluster_frames

def spectral_clustering(W, k, prefix, spectral_type, gamma_s, gamma_c):

    load_path = 'eigen_vec/{}_{}_sorted_egvecs_{:.5f}_{:.5f}.npy'.format(prefix, spectral_type, gamma_s, gamma_c)
    cluster_frames = []
    
    if os.path.exists(load_path):
        with open(load_path, 'rb') as f:
            eigen_vecs = np.load(f)
            H = eigen_vecs[:, 1:k+1].real
            
            cluster_frames = kmeans(H, k)
    else :
        data_size = W.shape[0]
        D = np.zeros((data_size, data_size))
        for i in range(data_size):
            D[i, i] = np.sum(W[i])
        
        L = D - W 

        if spectral_type == 'ratio-cut':

            eigen_vals, eigen_vecs = np.linalg.eig(L)
        
        elif spectral_type == 'normalized-cut':

            # D^-0.5 x L x D^-0.5 = I - D^-0.5 x W x D^-0.5
            L_sym = np.linalg.inv(D ** 0.5) @ L @ np.linalg.inv(D ** 0.5)
            eigen_vals, eigen_vecs = np.linalg.eig(L_sym)

        sorted_ind = np.argsort(eigen_vals)
        eigen_vecs = eigen_vecs[:, sorted_ind].real
        save_path = 'eigen_vec/{}_sorted_egvecs_{:.5f}_{:.5f}.npy'.format(spectral_type, gamma_s, gamma_c)
        with open(save_path, 'wb') as f:
            np.save(f, eigen_vecs)
            print(f'write {save_path} done.')

        H = eigen_vecs[:, 1:k+1]
        cluster_frames = kmeans(H, k)

    return cluster_frames

def save_eigenvec(W, prefix, gamma_s, gamma_c):
    data_size = W.shape[0]
    D = np.zeros((data_size, data_size))
    for i in range(data_size):
        D[i, i] = np.sum(W[i])
    
    L = D - W 

    eigen_vals, eigen_vecs = np.linalg.eig(L)
    sorted_ind = np.argsort(eigen_vals)
    eigen_vecs = eigen_vecs[:, sorted_ind]
    save_path = 'eigen_vec/{}_ratio-cut_sorted_egvecs_{:.5f}_{:.5f}.npy'.format(prefix, gamma_s, gamma_c)
    with open(save_path, 'wb') as f:
        np.save(f, eigen_vecs)
        print(f'write {save_path} done.')

    # D^-0.5 x L x D^-0.5 = I - D^-0.5 x W x D^-0.5
    L_sym = np.linalg.inv(D ** 0.5) @ L @ np.linalg.inv(D ** 0.5)
    eigen_vals, eigen_vecs = np.linalg.eig(L_sym)
    sorted_ind = np.argsort(eigen_vals)
    eigen_vecs = eigen_vecs[:, sorted_ind]
    save_path = 'eigen_vec/{}_normalized-cut_sorted_egvecs_{:.5f}_{:.5f}.npy'.format(prefix, gamma_s, gamma_c)
    with open(save_path, 'wb') as f:
        np.save(f, eigen_vecs)
        print(f'write {save_path} done.')
    

def main(args):
    paths = glob.glob(f'*.png')

    k = args.k
    gamma_c = args.gamma_c
    gamma_s = args.gamma_s
    clustering_type = {1:'kkmeans', 2:'spectral'}[args.ctype]
    spectral_type = {1:'ratio-cut', 2:'normalized-cut'}[args.stype]
    init_type = {1:'random', 2:'kmeans++'}[args.itype]

    palette = get_palette(k)
    for path in paths:
        print(f'[processing kernel K-mean of image {path}]')

        img = cv2.imread(path)
        img_resize = cv2.resize(img, (100, 100))

        # get_gram_matrix img data with respect to color and space
        rgb_flatten = get_flattened_imrbg(img_resize)
        loc_flatten = get_flattened_imloc(img_resize.shape)
        gram_mat = get_gram_matrix(rgb_flatten, loc_flatten, gamma_s=gamma_s, gamma_c=gamma_c)
        
        prefix = os.path.splitext(path)[0]
        
        if args.save_eigenvec :

            save_eigenvec(gram_mat, prefix, gamma_s=gamma_s, gamma_c=gamma_c)

        else :

            if clustering_type == 'kkmeans':

                # run kernel kmeans
                cluster_frames = kernel_kmeans(gram_mat, k)
                save_path = f'output/{prefix}_{clustering_type}_{k}_{init_type}_{gamma_s}_{gamma_c}.gif'

            elif clustering_type == 'spectral':

                # run spectral clustering
                cluster_frames = spectral_clustering(gram_mat, k, prefix, spectral_type, gamma_s, gamma_c)
                save_path = f'output/{prefix}_{clustering_type}_{spectral_type}_{k}_{init_type}_{gamma_s}_{gamma_c}.gif'

            else :

                raise Exception(f'clustering type error : {clustering_type}')
            
            visualize_clusters(save_path, cluster_frames, img_resize.shape, palette)
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ctype', '-c', type=int, default=1, help='the clustering type [1:kkmeans, 2:spectral]')
    parser.add_argument('--stype', '-s', type=int, default=1, help='type of spectral clustering [1:ratio_cut, 2:normalized_cut]')
    parser.add_argument('--itype', '-i', type=int, default=1, help='initialization methods [1:random, 2:kmeans++]')
    parser.add_argument('--gamma_c', '-gc', type=float, default=0.0001, help='gamma c for kernel')
    parser.add_argument('--gamma_s', '-gs', type=float, default=0.0001, help='gamma s for kernel')
    parser.add_argument('--k', '-k', type=int, default=3, help='k for clustering')
    parser.add_argument('--save_eigenvec', '-seg', action='store_true', default=False, help='save eigen vector only')
    args = parser.parse_args()
    main(args)