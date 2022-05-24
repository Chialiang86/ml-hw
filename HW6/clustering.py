import argparse
import numpy as np
import cv2
# import copy
import os
import glob

from utils import get_flattened_imloc, get_flattened_imrbg, get_gram_matrix, get_palette, \
                visualize_eigenspace, visualize_clusters

MAX_K = 10

def init_clusters(data, k, init_type):
    '''
    @param data: ndarray with shape (n, dim), input data or input feature
    @param k: int, number of clusters for kmeans initialization
    @param initType: string, ['random','kmeans++']
    @return: ndarray with shape (k, dim), 
    '''
    data_size = data.shape[0]
    data_dim = data.shape[1]

    centers = np.zeros((k, data_dim))
    if init_type == 'random':
        clusters = np.array([np.random.randint(0, k) for i in range(data_size)])
        for j in range(k):
            cond = np.where(clusters == j)
            centers[j] = np.mean(data[cond], axis=0)

    elif init_type == 'kmeans++':
        first_center = data[np.random.randint(data_size)] # randomly choose one as first center
        centers[0] = first_center
        for j in range(1, k):
            dist = np.array([min([np.sum((x - c)**2) for c in centers]) for x in data])
            dist /= np.sum(dist)
            cumulative_dist = np.cumsum(dist)
            r = np.random.rand()
            for jj, p in enumerate(cumulative_dist):
                if r < p:
                    i = jj
                    break
            
            centers[j] = data[i]

    return centers


def kmeans(data, k, init_type):
    '''
    @param data: ndarray with shape (n, dim), the input data or features
    @param k: int, number of clusters for kmeans initialization
    @param initType: string, ['random','kmeans++']
    @return: list of ndarray with shape (n, 1), a list of the clustering result for each iteration
    '''
    data_size = data.shape[0]
    
    # for computing the distance to each center
    dist_map = np.zeros((data_size, k))
    # initial centers
    means = init_clusters(data, k, init_type)
    # for comparing with new clustering
    cluster_old = -np.ones(data_size) # initialize as -1
    # for recording cluster result for each step 
    cluster_frames = []
    
    diff = np.inf
    iteration = 0
    max_iteration = 1000
    thresh = 10
    while diff > thresh and iteration < max_iteration:
        iteration += 1
        
        # E step -> update clusters with keeping centers unchanged
        for j in range(k):
            dist_map[:, j] = np.sum((data - means[j]) ** 2, axis=1)
        cluster = np.argmin(dist_map, axis=1)

        # M step -> update centers with keeping clusters unchanged
        for j in range(k):
            cond = np.where(cluster == j)
            means[j] = np.mean(data[cond], axis=0)

        # check if converge
        diff = np.sum(np.abs(cluster - cluster_old))
        cluster_old = cluster
        print(f'[kmeans iteration : {iteration}, diff : {diff}]')

        cluster_frames.append(cluster)
    
    return cluster_frames


# def kernel_kmeans(gram_mat, k, init_type):
    

    # # get the diagonal value of the gram matrix to form a new list 
    # gram_D = np.array([gram_mat[i, i] for i in range(data_size)])

    # # initialize cluster
    # cluster_old = np.array([np.random.randint(0, k) for i in range(data_size)])
    # dist_map = np.zeros((data_size, k))
    # indicator_map = np.zeros((data_size, k)) # orthogonal to each other
    # for j in range(k):
    #     cond = np.where(cluster_old == j)
    #     indicator_map[cond, j] = 1
    
    # cluster_frames = []
    # diff = np.inf
    # iteration = 0
    # thresh = 10
    # while diff > thresh:
    #     iteration += 1

    #     for j in range(k):
    #         Cj_num = np.sum(indicator_map[:,j]) # |Ck|
    #         # k(x_j, x_j)
    #         first_term = gram_D 
    #         # 2 / |Ck| * sum_n(alpha_kn * k(x_j, x_n))
    #         second_term = 2 / Cj_num * np.dot(gram_mat, indicator_map[:, j])
    #         # 1 / |Ck|^2 * sum_p(sum_q(alpha_kp * alpha_kq * k(x_p, x_q)))
    #         third_term = (1 / Cj_num**2) * np.dot(np.dot(indicator_map[:, j].T, gram_mat), indicator_map[:, j])
            
    #         dist_map[:, j] = first_term - second_term + third_term

    #     # assign to new cluster/indicator mat
    #     cluster = np.argmin(dist_map, axis=1)
    #     indicator_map = np.zeros((data_size, k))
    #     for j in range(k):
    #         cond = np.where(cluster == j)
    #         indicator_map[cond, j] = 1

    #     # count difference between current cluster and new cluster
    #     diff = np.sum(np.abs(cluster - cluster_old)) 
    #     cluster_old = copy.deepcopy(cluster)
    #     print(f'[kernel kmeans iteration : {iteration}, diff : {diff}]')

    #     cluster_frames.append(cluster)
    
    # return cluster_frames

def spectral_clustering(W, k, init_type, prefix, spectral_type, gamma_s, gamma_c):
    '''
    @param W: ndarray, gram matrix of the image data
    @param k: int, number of clusters
    @param initType: string, ['random','kmeans++'] for kmeans initialization
    @param prefix: string, for output file's directory
    @param spectral_type: string, ['ratio-cut','normalized-cut']
    @param gamma_s: float, for computing the spacial similarity kernel of image data
    @param gamma_c: float, for computing the color similarity kernel of image data
    @return: list of ndarray with shape (n, 1), a list of the clustering result for each iteration
    '''

    eigen_path = 'eigen_vec/{}_{}_sorted_eg_{:.5f}_{:.5f}.npz'.format(prefix, spectral_type, gamma_s, gamma_c)
    cluster_frames = []
    
    # the corresponding eigen vectors are exist -> don't need to compute from scratch
    if os.path.exists(eigen_path):
        # load the eigen vector of the corresponding matrix (Laplacian matrix or normalized Laplacian matrix)
        with open(eigen_path, 'rb') as f:
            eigen = np.load(f)
            # load first k eigen vectors
            U = eigen['eg_vec'][:, :k]
            # normalize each coordinate within 0->1
            sums=np.sqrt(np.sum(np.square(U),axis=1)).reshape(-1,1)
            T = U/sums
            
            # pass first k eigenspace of Laplacian matrix or normalized Laplacian matrix
            # as input data of kmeans
            cluster_frames = kmeans(T, k, init_type)
            
            # for visualizing the eigenspace
            if k == 3:
                final_cluster = cluster_frames[-1]
                title = f'Eigenspace {prefix} Using {spectral_type} (k=3, gamma_s={gamma_s}, gamma_c={gamma_c})'
                save_path = f'output/{prefix}/eigen_space/{spectral_type}_{init_type}_{gamma_s}_{gamma_c}.png'
                visualize_eigenspace(final_cluster, T, title, save_path)
    
    # the corresponding eigen vectors aren't exist -> need to compute from scratch
    else :
        data_size = W.shape[0]

        # setting Diagonal matrix
        D = np.zeros((data_size, data_size))
        for i in range(data_size):
            D[i, i] = np.sum(W[i])
        
        # graph laplacian matrix, where W is the kernel similarity matrix
        L = D - W 

        if spectral_type == 'ratio-cut':
            # just need the eigen vectors of the original graph laplacian matrix
            eigen_vals, eigen_vecs = np.linalg.eig(L)
        elif spectral_type == 'normalized-cut':
            # eigen vectors of the normalized graph laplacian matrix
            # L_sym = D^-0.5 x L x D^-0.5 = I - D^-0.5 x W x D^-0.5
            L_sym = np.linalg.inv(D ** 0.5) @ L @ np.linalg.inv(D ** 0.5)
            eigen_vals, eigen_vecs = np.linalg.eig(L_sym)

        # sort the eigen values as ascending order
        sorted_ind = np.argsort(eigen_vals)
        eigen_vecs = eigen_vecs[:, sorted_ind].real # just need the real part

        # load first k eigen vectors
        U = eigen_vecs[:, :k]
        # normalize each coordinate within 0->1
        sums=np.sqrt(np.sum(np.square(U),axis=1)).reshape(-1,1)
        T = U/sums
            
        # pass first k eigenspace of Laplacian matrix or normalized Laplacian matrix
        # as input data of kmeans
        cluster_frames = kmeans(T, k)
        
        # for visualizing the eigenspace
        if k == 3:
            final_cluster = cluster_frames[-1]
            title = f'Eigenspace of {prefix} Using {spectral_type} (k=3, gamma_s={gamma_s}, gamma_c={gamma_c})'
            save_path = f'output/{prefix}/eigen_space/{spectral_type}_{init_type}_{gamma_s}_{gamma_c}.png'
            visualize_eigenspace(final_cluster, T, title, save_path)

    return cluster_frames

def save_eigenvec(W, prefix, gamma_s, gamma_c):
    '''
    @param W: ndarray, gram matrix of the image data
    @param prefix: string, for output file's directory
    @param gamma_s: float, for computing the spacial similarity kernel of image data
    @param gamma_c: float, for computing the color similarity kernel of image data
    @return: list of ndarray, a list of the clustering result for each iteration
    '''
    data_size = W.shape[0]

    # setting Diagonal matrix
    D = np.zeros((data_size, data_size))
    for i in range(data_size):
        D[i, i] = np.sum(W[i])
    
    # graph laplacian matrix, where W is the kernel similarity matrix
    L = D - W 

    # for ratio cut
    eigen_vals, eigen_vecs = np.linalg.eig(L)
    sorted_ind = np.argsort(eigen_vals)
    eigen_vals = eigen_vals[sorted_ind].real
    eigen_vecs = eigen_vecs[:, sorted_ind].real

    # save first (MAX_K=10) eigen vectors
    eigen_vals_k = eigen_vals[:MAX_K]
    eigen_vecs_k = eigen_vecs[:, :MAX_K]
    save_path = 'eigen_vec/{}_ratio-cut_sorted_eg_{:.5f}_{:.5f}.npz'.format(prefix, gamma_s, gamma_c)
    with open(save_path, 'wb') as f:
        np.savez(f, eg_val=eigen_vals_k, eg_vec=eigen_vecs_k)
        print(f'write {save_path} done.')

    # for normalized cut
    # D^-0.5 x L x D^-0.5 = I - D^-0.5 x W x D^-0.5
    L_sym = np.linalg.inv(D ** 0.5) @ L @ np.linalg.inv(D ** 0.5)
    eigen_vals, eigen_vecs = np.linalg.eig(L_sym)
    sorted_ind = np.argsort(eigen_vals)
    eigen_vals = eigen_vals[sorted_ind].real
    eigen_vecs = eigen_vecs[:, sorted_ind].real

    # save first (MAX_K=10) eigen vectors
    eigen_vals_k = eigen_vals[:MAX_K]
    eigen_vecs_k = eigen_vecs[:, :MAX_K]
    save_path = 'eigen_vec/{}_normalized-cut_sorted_eg_{:.5f}_{:.5f}.npz'.format(prefix, gamma_s, gamma_c)
    with open(save_path, 'wb') as f:
        np.savez(f, eg_val=eigen_vals_k, eg_vec=eigen_vecs_k)
        print(f'write {save_path} done.')

def main(args):
    paths = glob.glob(f'*.png')

    k = args.k
    assert k < MAX_K and k > 0, f'k should be in [0, {MAX_K}]'

    gamma_c = args.gamma_c
    gamma_s = args.gamma_s
    clustering_type = {1:'kkmeans', 2:'spectral'}[args.ctype]
    spectral_type = {1:'ratio-cut', 2:'normalized-cut'}[args.stype]
    init_type = {1:'random', 2:'kmeans++'}[args.itype]

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
                cluster_frames = kmeans(gram_mat, k, init_type)
                save_path = f'output/{prefix}/gif/{clustering_type}_{init_type}_{k}_{gamma_s}_{gamma_c}.gif'
            elif clustering_type == 'spectral':
                # run spectral clustering
                cluster_frames = spectral_clustering(gram_mat, k, init_type, prefix, spectral_type, gamma_s, gamma_c)
                save_path = f'output/{prefix}/gif/{spectral_type}_{init_type}_{k}_{gamma_s}_{gamma_c}.gif'
            else :
                raise Exception(f'clustering type error : {clustering_type}')
            
            visualize_clusters(save_path, cluster_frames, img_resize.shape)
        

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