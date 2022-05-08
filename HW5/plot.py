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
        
    elif len(data.shape) == 2:
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.title(f'RBF result')
        plt.imshow(data.T, cmap='cool', interpolation='nearest', aspect='auto')
        plt.xlabel('gamma')
        plt.ylabel('c')
        plt.xticks(np.arange(len(x_label)), labels=x_label)
        plt.yticks(np.arange(len(y_label)), labels=y_label)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(len(x_label)):
            for j in range(len(y_label)):
                text = plt.text(j, i, data[i, j], ha='center', va='center', color='w')
        
        plt.savefig(f'task2.2_RBF.png')

if __name__=="__main__":
    gamma_default = 0.001
    coef_default = 0
    degree_default = 3
    c_default = 1


    degree_list = [degree_default - 1, degree_default, degree_default + 1]
    gamma_list = [gamma_default / 1000, gamma_default / 100, gamma_default / 10, gamma_default, 
                  gamma_default * 10, gamma_default * 100, gamma_default * 1000]
    coef_list = [coef_default - 100, coef_default - 10, coef_default - 1, coef_default - 0.1, coef_default,
                 coef_default + 0.1, coef_default + 1, coef_default + 10, coef_default + 100]
    polynomial_acc_map = np.random.random((len(degree_list), len(gamma_list), len(coef_list)))
    plot_res(polynomial_acc_map, degree_list, gamma_list, coef_list)


    # for RBF exp(-gamma*|u-v|^2)
    gamma_list = [gamma_default / 1000, gamma_default / 100, gamma_default / 10, gamma_default, 
                  gamma_default * 10, gamma_default * 100, gamma_default * 1000]
    c_list = [ c_default / 100, c_default / 10, c_default,
               c_default * 10, c_default * 100]
    RBF_acc_map = np.random.random((len(gamma_list), len(c_list)))
    plot_res(RBF_acc_map, gamma_list, c_list, None)
