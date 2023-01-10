import os
import time
import shutil
import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F
from scipy.signal import stft

import pandas as pd
from tqdm import tqdm

from util import load_mat, save_mat

import random


class Dataset_proecess():
    def __init__(self,
                 dataset_path,
                 save_path):
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.train_list_path = os.path.join(save_path, 'train_list.csv')
        self.test_list_path = os.path.join(save_path, 'test_list.csv')


    def split_train_test(self):
        file_list = os.listdir(self.dataset_path)
        train_list = []
        test_list = []
        with tqdm(file_list) as file_list:
            for file_name in file_list:
                file_n, _  = file_name.split('.')
                data_info, num = file_n.split('_')
                location, person, action = data_info.split('-')

                if num in ['05','10']:
                    test_list.append([file_n, action, location, person])
                else:
                    train_list.append([file_n, action, location, person])

        file_list.write(f'train_list: {len(train_list)},  test_list: {len(test_list)}')
        file_list.close()

        df_train = pd.DataFrame(train_list, columns=['file', 'label', 'location', 'person'])
        df_test = pd.DataFrame(test_list, columns=['file', 'label', 'location', 'person'])

        df_train.to_csv(self.train_list_path, index=False)
        df_test.to_csv(self.test_list_path, index=False)

        self.normalize_mean_std(test_list)

    def save_data_h5(self, file_list, save_path):
        pass

    def normalize_mean_std(self, data_list):

        def _load_mat_data(data_path):
            data = scio.loadmat(data_path)['absMat']
            C, D, N = data.shape
            data = np.transpose(data, (2, 0, 1)).reshape((C * N, D))
            return data

        def _get_mean_std(data):
            print(data.shape)
            pass

        loc_list = self.read_file_list_loc(data_list)

        for index, loc in enumerate(loc_list):
            with tqdm(loc) as loc:
                amp_list = []
                for file in loc:
                    loc.set_description(f'location: {index}')
                    data = _load_mat_data(file)
                    amp_list.append(np.expand_dims(data, 0))
                data = np.concatenate(amp_list, axis=0)
                # loc.write(data.shape)
                loc.close()
            _get_mean_std(data)



    def read_file_list_loc(self, data_list):
        """
            返回按位置分好的 mat 文件的路径
        """


        loc_list = [[] for _ in range(8)]

        for file_n, action, location, person in data_list:
            data_path = os.path.join(self.dataset_path, f'{file_n}.mat')
            loc_list[int(location)-1].append(data_path)

        return loc_list
            # data = _load_mat_data(data_path)






if __name__ == '__main__':

    """
        setting
    """
    if_split        = False
    if_normalize    = False
    if_downsample   = False
    downsample_factor = 5

    """
        路径相关：
    """
    dataset_path    = '/home/lanbo/dataset/wifi_violence/'
    save_path       = '/home/lanbo/dataset/wifi_violence_processed_loc/'
    train_path      = os.path.join(save_path, 'train')
    train_list_path = os.path.join(save_path, 'train_list.csv')
    test_path       = os.path.join(save_path, 'test')
    test_list_path  = os.path.join(save_path, 'test_list.csv')

    mean_std_path   = os.path.join('/home/lanbo/dataset/wifi_violence_processed', 'mean_std.h5')

    save_path = os.path.join(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    """
        check_data_list
    """
    index_list = [5, 1000, 2000]

    '''
        数据集划分
    '''
    dataset_process = Dataset_proecess(dataset_path, save_path)
    dataset_process.split_train_test()