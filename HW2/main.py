import numpy as np
import struct
import argparse 

fname_train_img = 'train-images.idx3-ubyte'
fname_train_label = 'train-labels.idx1-ubyte'
fname_t10k_img = 't10k-images.idx3-ubyte'
fname_t10k_label = 't10k-labels.idx1-ubyte'

def main():
    with open(fname_t10k_img, 'rb') as f_t10k_img:
        magic = int.from_bytes(f_t10k_img.read(4), 'big')
        num_of_img = int.from_bytes(f_t10k_img.read(4), 'big')
        rows = int.from_bytes(f_t10k_img.read(4), 'big')
        cols = int.from_bytes(f_t10k_img.read(4), 'big')
        print(f'num of imgs : {num_of_img}, rows = {rows} cols = {cols}')
        
    
    # with open(fname_t10k_label, 'r') as f_t10k_label:
    #     lines = f_t10k_label.readlines()
    #     print(f'lines : {len(lines)}')
    
    # with open(fname_train_img, 'r') as f_train_img:
    #     lines = f_train_img.readlines()
    #     print(f'lines : {len(lines)}')

    # with open(fname_train_label, 'r') as f_train_label:
    #     lines = f_train_label.readlines()
    #     print(f'lines : {len(lines)}')

if __name__=="__main__":
    main()