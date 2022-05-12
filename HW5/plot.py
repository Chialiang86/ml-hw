import numpy as np
import matplotlib.pyplot as plt


def plot_res(data, x_label, y_label, z_label):
    if len(data.shape) == 3:
        for i in range(data.shape[0]):
            fig, ax = plt.subplots(figsize=(10, 10))
            plt.title(f'polynomial result : degree = {x_label[i]}')
            plt.imshow(data[i].T, cmap='cool', interpolation='nearest', aspect='auto')
            plt.xlabel('gamma')
            plt.ylabel('coef')
            plt.xticks(np.arange(len(y_label)), labels=y_label)
            plt.yticks(np.arange(len(z_label)), labels=z_label)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            for j in range(len(y_label)):
                for k in range(len(z_label)):
                    text = plt.text(k, j, data[i, j, k], ha='center', va='center', color='w')
            plt.savefig(f'task2.2_polynomial_{x_label[i]}.png')

if __name__=="__main__":
    file = open('task2.2_out.txt')

    lines = file.readlines()

    



    plot_res(RBF_acc_map, gamma_list, c_list, None)
