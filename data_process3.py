import os
import time
import shutil
import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F
from scipy.signal import stft

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from util import load_mat, save_mat

import random


class Dataset_proecess():
    def __init__(self,
                 dataset_path,
                 save_path,
                 nor_type = 'mean_std',
                 down_factor = 5):
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.train_list_path = os.path.join(save_path, 'train_list.csv')
        self.test_list_path = os.path.join(save_path, 'test_list.csv')
        self.train_data_path = os.path.join(save_path, 'train')
        self.test_data_path = os.path.join(save_path, 'test')
        self.mean_std_path = os.path.join(save_path, 'mean_std')
        self.path_check(self.train_data_path)
        self.path_check(self.test_data_path)
        self.path_check(self.mean_std_path)
        self.nor_type = nor_type
        self.down_factor = down_factor

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


        if not self._check_mean_std():
            self.normalize_mean_std(train_list)
        else:
            print('use exist mean_std')

        self.save_data_mean_std(train_list, self.train_data_path, 'train', self.nor_type)
        self.save_data_mean_std(test_list, self.test_data_path, 'test', self.nor_type)

    def save_data_mean_std(self, data_list, save_path, name, nor_type):

        def _load_mat_data(data_path):
            data = scio.loadmat(data_path)['absMat']
            C, D, N = data.shape
            data = np.transpose(data, (2, 0, 1)).reshape((C * N, D))
            return data

        def _normalize(data, mean, std):
            return (data - mean) / std

        def _normalize_lin(data, mean, max, min):
            return (data - mean) / (max - min)

        def _downsample(data):
            n_channel, seq_len = data.shape
            data = torch.unsqueeze(torch.Tensor(data), dim=0)
            data = F.interpolate(data, seq_len // self.down_factor, mode='linear')
            data = torch.squeeze(data).numpy()
            return data

        loc_list = self.read_file_list_loc(data_list)

        check_data_list = [[] for _ in range(8)]

        for index, loc in enumerate(loc_list):
            mean_std = load_mat(os.path.join(self.mean_std_path, f'mean_std_{index}.h5'))
            with tqdm(loc) as loc:
                for i, file in enumerate(loc):
                    loc.set_description(f'nor_location: {index}')
                    data = _load_mat_data(file['data_path'])
                    if i == 0:
                        check_data_list[index].append(data)
                    if nor_type == 'mean_std':
                        data = _normalize(data, mean_std['mean'], mean_std['std'])
                    elif nor_type == 'linear':
                        data = _normalize_lin(data, mean_std['mean'], mean_std['max'], mean_std['min'])
                    if i == 0:
                        check_data_list[index].append(data)
                    data = _downsample(data)
                    if i == 0:
                        check_data_list[index].append(data)

                    save_mat(os.path.join(save_path, f'{file["file"]}.h5'),
                             {
                                 'amp': data,
                                 'label': int(file["label"])
                             })
        self.check_data(check_data_list, name)

    # def save_data_mean_lin(self, data_list, save_path, name):

    def check_data(self, check_data_list, name):
        plt.figure(figsize=(15,15))
        j = 1
        for loc in check_data_list:
            for data in loc:
                plt.subplot(8,3,j)
                for i in range(30):
                    plt.plot(data[i,:])
                j = j + 1
        f = plt.gcf()
        f.savefig(os.path.join(self.save_path, f'{name}.png'))

        plt.show()
        f.clear()  # 释放内存

    def split_file(self, file):
        file_n, _ = file.split('.')
        data_info, num = file_n.split('_')
        location, person, action = data_info.split('-')
        return file_n, location, person, action

    def normalize_mean_std(self, data_list):

        def _load_mat_data(data_path):
            data = scio.loadmat(data_path)['absMat']
            C, D, N = data.shape
            data = np.transpose(data, (2, 0, 1)).reshape((C * N, D))
            return data

        def _get_mean_std(data):
            channel = data.shape[1]
            mean = np.mean(data.transpose(1, 0, 2).reshape(channel, -1), axis=-1)
            mean = np.expand_dims(mean, 1)
            std = np.std(data.transpose(1, 0, 2).reshape(channel, -1), axis=-1)
            std = np.expand_dims(std, 1)
            max = np.max(data.transpose(1, 0, 2).reshape(channel, -1), axis=-1)
            max = np.expand_dims(max, 1)
            min = np.min(data.transpose(1, 0, 2).reshape(channel, -1), axis=-1)
            min = np.expand_dims(min, 1)
            return mean, std, max, min

        loc_list = self.read_file_list_loc(data_list)

        for index, loc in enumerate(loc_list):
            with tqdm(loc) as loc:
                amp_list = []
                for file in loc:
                    loc.set_description(f'location: {index}')
                    data = _load_mat_data(file['data_path'])
                    amp_list.append(np.expand_dims(data, 0))
                data = np.concatenate(amp_list, axis=0)
                # loc.write(data.shape)
                loc.close()
            mean, std, max, min = _get_mean_std(data)
            save_mat(os.path.join(self.mean_std_path, f'mean_std_{index}.h5'),
                     {
                         'mean': mean,
                         'std': std,
                         'max': max,
                         'min': min
                     })

    def _check_mean_std(self):
        for index in range(8):
            mean_std_path = os.path.join(self.mean_std_path, f'mean_std_{index}.h5')
            if os.path.exists(mean_std_path):
                mean_std = load_mat(mean_std_path)
                print(mean_std['mean'].shape, mean_std['std'].shape, mean_std['max'], mean_std['min'])
                if mean_std['mean'].shape[0] != mean_std['std'].shape[0]:
                    return False
            else:
                return False
        return True



    def read_file_list_loc(self, data_list):
        """
            返回按位置分好的 mat 文件的路径
        """
        loc_list = [[] for _ in range(8)]
        for file_n, action, location, person in data_list:
            data_path = os.path.join(self.dataset_path, f'{file_n}.mat')
            loc_list[int(location)-1].append(
                {
                    'data_path': data_path,
                    'file': file_n,
                    'label': action
                })
        return loc_list
            # data = _load_mat_data(data_path)


    def path_check(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path





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
    # dataset_path    = '/home/lanbo/dataset/wifi_violence/'
    # save_path       = '/home/lanbo/dataset/wifi_violence_processed_loc/'

    dataset_path    = 'wifi_partition_data_abs'
    save_path       = 'dataset2'

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