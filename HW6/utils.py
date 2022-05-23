from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imageio

MAX_K = 10

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

def visualize_eigenspace(final_cluster, T, title, save_path):

        cls0ind = np.where(final_cluster == 0)
        cls1ind = np.where(final_cluster == 1)
        cls2ind = np.where(final_cluster == 2)

        # Creating figure
        fig = plt.figure(figsize = (10, 10))
        ax = plt.axes(projection ="3d")
        ax.set_title(title)
        ax.scatter3D(T[cls0ind, 0], T[cls0ind, 1], T[cls0ind, 2], color='r', label='cls 0')
        ax.scatter3D(T[cls1ind, 0], T[cls1ind, 1], T[cls1ind, 2], color='g', label='cls 1')
        ax.scatter3D(T[cls2ind, 0], T[cls2ind, 1], T[cls2ind, 2], color='b', label='cls 2')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.savefig(save_path)

def visualize_clusters(save_path, clusters, shape):
    assert len(shape) == 3, f'invalid shape size: {shape}'

    gif_frames = []
    for cluster in clusters:
        cluster_map = ((np.reshape(cluster, shape[:2]) * 50) % 256).astype(np.uint8)
        gif_frame = cv2.applyColorMap(cluster_map, cv2.COLORMAP_HSV)
        gif_frames.append(gif_frame)

    imageio.mimsave(save_path, gif_frames, fps=5)