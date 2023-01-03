import os
import os.path

import numpy as np
import scipy.io as scio


# def load_mat_data(data_path: os.path):



def load_data(root: os.path):
    file_list = os.listdir(root)

    for i, file_name in enumerate(file_list):
        # location, person, action, num = file_name.split('_')
        file_name, _ = file_name.split('.')
        data_info, num = file_name.split('_')
        location, person, action = data_info.split('-')
        print(i, '===', location, person, action, num)

if __name__ == '__main__':
    dataset_path = 'D:\study\dataset\wifi-partition-data-abs'
    dataset_path = os.path.join(dataset_path, 'wifi_partition_data_abs')

    # print(dataset_path)
    # load_data(dataset_path)


    data_path = os.path.join(dataset_path, '1-12-1_01.mat')
    data = scio.loadmat(data_path)['absMat']
    C, D, N = data.shape
    print(data.shape)
    data_new = np.transpose(data, (2,0,1)).reshape((C*N, D))
    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(30):
        plt.subplot(2,1,1)
        plt.plot(data[i,:,1])
        plt.subplot(2, 1, 2)
        plt.plot(data_new[i+30,:])
    plt.show()
    print(data_new.shape)