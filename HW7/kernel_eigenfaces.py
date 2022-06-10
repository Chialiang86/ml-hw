import argparse
import numpy as np
from PIL import Image
import scipy.spatial.distance
import cv2
import csv
import matplotlib.pyplot as plt
import os

from torch import argsort 

IMSIZE = (97, 115)

def read_data(dir, imsize=(97, 115)):
    paths = os.listdir(dir)
    images = []
    labels = []
    fnames = []
    for path in paths:
        img = Image.open(os.path.join(dir, path))
        img = img.resize(imsize, Image.ANTIALIAS)
        img = np.asarray(img).flatten()
        images.append(img)
        labels.append(int(path.split('.')[0][-2:]) - 1)
        fnames.append(os.path.join(dir, path))

    return np.array(images), np.array(labels), np.array(fnames)

def pca(x, k=None):
    x_mean = np.mean(x, axis=0)
    x_centerize = x - x_mean

    cov = x_centerize @ x_centerize.T
    eg_vals, eg_vecs = np.linalg.eig(cov)

    ind = np.argsort(-eg_vals)
    eg_vals = eg_vals[ind]
    eg_vecs = eg_vecs[ind]
    
    if k is None:
        for i in range(len(eg_vals)):
            if eg_vals[i] < 0:
                k = i
                break 
    
    eg_vecs = eg_vecs[:,:k].real
    eg_vecs_sum = np.sum(eg_vecs, axis=0)
    eg_vec_norm = eg_vecs / eg_vecs_sum

    W = x_centerize.T @ eg_vec_norm

    return W, x_mean

# def show_eg_faces(W, input_fnames, output_dir, imsize=(97, 115)):

    
#     plt.clf()
#     for i in range(5):
#         for j in range(5):
#             idx = i * 5 + j
#             plt.subplot(5, 5, idx + 1)
#             plt.imshow(W[:, idx].reshape(imsize), cmap='gray')
#             plt.axis('off')
#     plt.savefig(os.path.join(output_dir, '25.png'))

#     for i in range(W.shape[1]):
#         plt.clf()
#         plt.title(f'{title}_{i + 1}')
#         plt.imshow(W[:, i].reshape(imsize), cmap='gray')
#         plt.savefig(f'./{folder}/{title}/{title}_{i + 1}.png')
    
#     plt.clf()
#     for i in range(2):
#         for j in range(5):
#             idx = i * 5 + j
#             plt.subplot(2, 5, idx + 1)
#             plt.imshow(reconstruction[idx].reshape(SHAPE[::-1]), cmap='gray')
#             plt.axis('off')
#     plt.savefig(f'./{folder}/reconstruction.png')

#     for i in range(reconstruction.shape[0]):
#         plt.clf()
#         plt.title(target_filename[i])
#         plt.imshow(reconstruction[i].reshape(SHAPE[::-1]), cmap='gray')
#         plt.savefig(f'./{folder}/{target_filename[i]}.png')


def main(args):
    training_dir = 'Yale_Face_Database/Training'
    testing_dir = 'Yale_Face_Database/Testing'
    training_images, training_labels, fnames = read_data(training_dir)

    assert 1 <= args.task and args.task <= 3, 'task num should be 1,2,3'

    if args.task == 1:
        randind = np.random.choice(len(fnames), 10)
        training_images_chosen = training_images[randind]
        fnames_chosen = fnames[randind]
        eg_faces, x_mean = pca(training_images, 25)
        print(eg_faces.shape)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', type=int, default=1)
    args = parser.parse_args()
    main(args)