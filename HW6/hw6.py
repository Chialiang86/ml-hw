from cv2 import kmeans
import numpy as np
import cv2
from scipy.spatial.distance import cdist
from sklearn import cluster

def kernelize(img, gamma_s=1, gamma_c=10):
    img_flatten = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    c_x = cdist(img_flatten, img_flatten) ** 2
    
    (xr, xc) = np.meshgrid(range(img.shape[0]), range(img.shape[1]), indexing='ij')
    xr = xr.flatten()
    xc = xc.flatten()
    xrc = np.vstack((xr, xc)).T
    s_x = cdist(xrc, xrc) ** 2

    # gram matrix
    k = np.exp(-gamma_s * s_x) + np.exp(-gamma_c * c_x)

    return k 

def kernel_kmeans(img, k):
    data_size = img.shape[0] * img.shape[1]
    
    # kernelize img data with respect to color and space
    kmeans = np.array([np.random.randint(0, data_size) for i in range(k)])

    r = np.zeros((data_size, data_size))

    # initialize cluster
    img_kernel = kernelize(img, 1, 10)
    cluster = np.zeros((1, data_size))




def main():
    paths = ['image1.png', 'image2.png']
    img1 = cv2.imread(paths[0])
    img2 = cv2.imread(paths[1])
    kernel_kmeans(img1, 2)

    return 0

if __name__=="__main__":
    main()