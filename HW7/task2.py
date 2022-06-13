import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from PIL import Image
import imageio


def h_beta(data: np.ndarray, beta: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """
    # Compute P-row and corresponding perplexity
    p = np.exp(-data.copy() * beta)
    sum_p = sum(p)
    h = np.log(sum_p) + beta * np.sum(data * p) / sum_p
    p = p / sum_p

    return h, p


def x2p(matrix_x: np.ndarray, tol: float = 1e-5, perplexity: float = 20.0) -> np.ndarray:
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """
    # Initialize some variables
    print('Computing pairwise distances')
    n, d = matrix_x.shape
    sum_x = np.sum(np.square(matrix_x), 1)
    d = np.add(np.add(-2 * np.dot(matrix_x, matrix_x.T), sum_x).T, sum_x)
    p = np.zeros((n, n))
    beta = np.ones((n, 1))
    log_u = np.log(perplexity)

    # Loop over all data points
    for i in range(n):
        # Print progress
        if i % 500 == 0:
            print(f'Computing P-values for point {i} of {n}...')

        # Compute the Gaussian kernel and entropy for the current precision
        beta_min = -np.inf
        beta_max = np.inf
        d_i = d[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
        h, this_p = h_beta(d_i, beta[i])

        # Evaluate whether the perplexity is within tolerance
        h_diff = h - log_u
        tries = 0
        while np.abs(h_diff) > tol and tries < 50:
            # If not, increase or decrease precision
            if h_diff > 0:
                beta_min = beta[i].copy()
                if beta_max == np.inf or beta_max == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + beta_max) / 2.
            else:
                beta_max = beta[i].copy()
                if beta_min == np.inf or beta_min == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + beta_min) / 2.

            # Recompute the values
            h, this_p = h_beta(d_i, beta[i])
            h_diff = h - log_u
            tries += 1

        # Set the final row of P
        p[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = this_p

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return p


def pca(matrix_x: np.ndarray, no_dims: int = 50) -> np.ndarray:
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """
    print('Preprocessing the data using PCA')
    n, d = matrix_x.shape
    difference = matrix_x - np.tile(np.mean(matrix_x, 0), (n, 1))
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(difference.T, difference))
    matrix_y = np.dot(difference, eigenvectors[:, 0:no_dims])

    return matrix_y

def sne(images: np.ndarray, labels: np.ndarray, mode: str, no_dims: int = 2, initial_dims: int = 50,
        perplexity: float = 20.0) -> np.ndarray:
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    matrix_x = pca(images, initial_dims).real
    n, d = matrix_x.shape
    max_iter = 1000
    momentum = 0.8
    eta = 500
    min_gain = 0.01
    ret_y = np.random.randn(n, no_dims)
    dy = np.zeros((n, no_dims))
    i_y = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # List storing images of clustering state
    gif_frames = []

    # Compute P-values
    p = x2p(matrix_x, 1e-5, perplexity)
    p = 4 * (p + np.transpose(p)) / np.sum(p)
    p = np.maximum(p, 1e-12) # promise to >= 0

    err_old = 10000
    # Run iters
    for iter in range(max_iter):

        # Compute pairwise affinities
        yij_square = np.sum(np.square(ret_y), 1) # N x 1
        yij_neg2 = -2. * np.dot(ret_y, ret_y.T) # N x N
        q_dividend = np.zeros((n, n))
        if mode == 't-SNE':
            # t-SNE
            q_dividend = 1. / (1. + np.add(np.add(yij_neg2, yij_square).T, yij_square))
        elif mode == 'symmetric-SNE':
            # symmetric SNE
            q_dividend = np.exp(-1. * np.add(np.add(yij_neg2, yij_square).T, yij_square))
        q_dividend[range(n), range(n)] = 0.
        q = q_dividend / np.sum(q_dividend)
        q = np.maximum(q, 1e-12)

        # Compute gradient
        pq = p - q
        if mode == 't-SNE':
            # t-SNE
            for i in range(n):
                # 4 x sum( (pij - qij) x (1 + || yi - yj ||)^(-1) x (yi - yj) )
                dy[i, :] = np.sum(np.tile(pq[:, i] * q_dividend[:, i], (no_dims, 1)).T * (ret_y[i, :] - ret_y), 0)
        elif mode == 'symmetric-SNE':
            # symmetric SNE
            for i in range(n):
                # 2 x sum( (pij - qij) x (yi - yj) )
                dy[i, :] = np.sum(np.tile(pq[:, i], (no_dims, 1)).T * (ret_y[i, :] - ret_y), 0)

        # Perform the update
        if iter < 20:
            momentum = 0.5
        elif iter < 200:
            momentum = 0.7
        else:
            momentum = 0.9
        gains = (gains + 0.2) * ((dy > 0.) != (i_y > 0.)) + \
                (gains * 0.8) * ((dy > 0.) == (i_y > 0.))
        gains[gains < min_gain] = min_gain
        i_y = momentum * i_y - eta * (gains * dy)
        ret_y = ret_y + i_y
        ret_y = ret_y - np.tile(np.mean(ret_y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            err = np.sum(p * np.log(p / q))
            print(f'Iteration {iter + 1}: error is {err}, diff is {err_old - err}...')
            if np.abs(err_old - err) < 0.00000001:
                break
            err_old = err

            # get scatter
            plt.clf()
            plt.scatter(ret_y[:, 0], ret_y[:, 1], 10, labels)
            plt.title(f'{mode}, perplexity = {perplexity}')
            plt.tight_layout()
            canvas = plt.get_current_fig_manager().canvas
            canvas.draw()
            img = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
            gif_frames.append(img)

        # Stop lying about P-values
        if iter == 100:
            p = p / 4.

    # Return solution
    return ret_y, p, q, gif_frames

def main(args):

    # Parse arguments
    image_file = 'tsne_python/mnist2500_X.txt'
    label_file = 'tsne_python/mnist2500_labels.txt'
    
    mode = None
    mode = {0:'t-SNE', 1:'symmetric-SNE'}[args.mode]
    # a measure of the effective number of neighbors
    perplexity = args.perplexity

    print(perplexity)

    # Read data
    x = np.loadtxt(image_file)
    label = np.loadtxt(label_file)

    y_ret, p, q, gif_frames = sne(x, label, mode, 2, 50, perplexity)
    
    # plot 2D reduction results
    plt.figure(2)
    plt.scatter(y_ret[:, 0], y_ret[:, 1], 10, label)
    plt.title(f'{mode} perplexity = {perplexity}')
    plt.tight_layout()
    plt.savefig(f'output/task2/{mode}_{perplexity}.png')

    # Plot pairwise similarities in high-dimensional space and low-dimensional space
    index = np.argsort(label)
    plt.clf()
    plt.figure(figsize=(10, 5))

    log_p = np.log(p) # for plotting p
    sorted_p = log_p[index][:, index]
    plt.subplot(121)
    img = plt.imshow(sorted_p, cmap='Greens', vmin=np.min(log_p), vmax=np.max(log_p))
    plt.colorbar(img)
    plt.title('High Dimensional')
    
    log_q = np.log(q) # for plotting q
    sorted_q = log_q[index][:, index]
    plt.subplot(122)
    img = plt.imshow(sorted_q, cmap='Greens', vmin=np.min(log_q), vmax=np.max(log_q))
    plt.colorbar(img)
    plt.title('Low Dimensional')
    plt.savefig(f'output/task2/{mode}_{perplexity}_highlow.png')
    plt.tight_layout()

    # Save gif
    filename = f'output/task2/{mode}_{perplexity}.gif'
    gif_frames[0].save(filename,
               save_all=True, append_images=gif_frames[1:], optimize=False, duration=200, loop=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=int, default=0, help='mode (0:t-SNE, 1:symmetric-SNE)')
    parser.add_argument('--perplexity', '-p', type=float, default=20, help='perplexity of SNE, default 20')
    args = parser.parse_args()
    main(args)