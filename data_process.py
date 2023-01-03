import os
import time
import shutil
import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F
from scipy.signal import stft
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

from util import load_mat, save_mat

import random


def split_train_test(dataset_path: os.path, save_path, rand = True, train_ratio = 0.8):
    file_list = os.listdir(dataset_path)
    class_list = [[] for _ in range(7)]

    for file_name in file_list:
        file_n, _ = file_name.split('.')
        data_info, num = file_n.split('_')
        location, person, action = data_info.split('-')
        # file = f'{location}_{person}_{action}_{num}'
        class_list[int(action)-1].append([file_n, action])

    train_list = []
    test_list = []

    for i, n_class in enumerate(class_list):
        train_num = int(train_ratio * len(n_class))
        print(f'class: {i}  class_len: {len(n_class)}  train_len: {train_num}   test_len: {len(n_class)-train_num}')
        if rand:
            random.shuffle(n_class)
        train_list.extend(n_class[:train_num])
        test_list.extend(n_class[train_num:])

    print('='*50)
    print(f'train_list: {len(train_list)},  test_list: {len(test_list)}')

    df_train = pd.DataFrame(train_list, columns=['file','label'])
    df_test = pd.DataFrame(test_list, columns=['file','label'])
    df_train.to_csv(os.path.join(save_path, 'train_list.csv'), index=False)
    df_test.to_csv(os.path.join(save_path, 'test_list.csv'), index=False)

    train_path = os.path.join(save_path, 'train')
    test_path = os.path.join(save_path, 'test')


    save_data_h5(test_list, test_path, dataset_path)


def save_data_h5(file_list, save_path, dataset_path):

    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    os.makedirs(save_path)

    for file, label in tqdm(file_list):

        data = scio.loadmat(os.path.join(dataset_path, file))['absMat']
        C, D, N = data.shape
        data = np.transpose(data, (2, 0, 1)).reshape((C * N, D))

        save_data = {'amp': data,
                     'label': int(label)}

        file_path = os.path.join(save_path, f'{file}.h5')

        save_mat(file_path, save_data)


def _normalize(train_data, test_data):
    def get_mean_std(data):
        channel = data.shape[1]
        mean = np.mean(data.transpose(1, 0, 2).reshape(channel, -1), axis=-1)
        mean = np.expand_dims(mean, 0)
        mean = np.expand_dims(mean, 2)
        std = np.std(data.transpose(1, 0, 2).reshape(channel, -1), axis=-1)
        std = np.expand_dims(std, 0)
        std = np.expand_dims(std, 2)
        return mean, std

    def normalize(data, mean, std):
        return (data - mean) / std

    amp_mean, amp_std = get_mean_std(train_data['data'])
    train_data['data'] = normalize(train_data['data'], amp_mean, amp_std)
    test_data['data'] = normalize(test_data['data'], amp_mean, amp_std)


def normalize_data(data_path: os.path):
    # train_data = scio.loadmat(os.path.join(datasource_path, 'train_dataset_%d.mat' % index))
    # test_data = scio.loadmat(os.path.join(datasource_path, 'test_dataset_%d.mat' % index))

    # TODO: 这里不对，要改
    train_data = load_mat(data_path)
    test_data = load_mat(data_path)

    _normalize(train_data, test_data)

    # scio.savemat(os.path.join(datasource_path, 'train_dataset_%d.mat' % index), train_data)
    # scio.savemat(os.path.join(datasource_path, 'test_dataset_%d.mat' % index), test_data)
    save_mat(data_path, train_data)
    save_mat(data_path, test_data)





def downsample_train_test(datasource_path: os.path, index: int, downsample_factor: int = 10):
    # train_data = scio.loadmat(os.path.join(datasource_path, 'train_dataset_%d.mat' % index))
    # test_data = scio.loadmat(os.path.join(datasource_path, 'test_dataset_%d.mat' % index))
    train_data = load_mat(os.path.join(datasource_path, 'train_dataset_%d.h5' % index))
    test_data = load_mat(os.path.join(datasource_path, 'test_dataset_%d.h5' % index))

    seq_len = train_data['data'].shape[2]
    assert seq_len == test_data['data'].shape[2]

    train_data['data'] = torch.Tensor(train_data['data'])
    train_data['data'] = F.interpolate(train_data['data'], seq_len // downsample_factor, mode='linear',
                                       align_corners=True)
    train_data['data'] = train_data['data'].numpy()

    test_data['data'] = torch.Tensor(test_data['data'])
    test_data['data'] = F.interpolate(test_data['data'], seq_len // downsample_factor, mode='linear',
                                      align_corners=True)
    test_data['data'] = test_data['data'].numpy()

    # scio.savemat(os.path.join(datasource_path, 'train_dataset_%d.mat' % index), train_data)
    # scio.savemat(os.path.join(datasource_path, 'test_dataset_%d.mat' % index), test_data)
    save_mat(os.path.join(datasource_path, 'train_dataset_%d.h5' % index), train_data)
    save_mat(os.path.join(datasource_path, 'test_dataset_%d.h5' % index), test_data)


def check_data(list_path, data_path):
    df = pd.read_csv(list_path)
    import matplotlib.pyplot as plt

    for index, row in df.iterrows():
        data = load_mat(os.path.join(data_path, f'{row["file"]}.h5'))
        if data['label'] == row['label']:
            print(data['amp'].shape, '-', data['label'], '-', row['label'])
        else:
            print('error')

    plt.figure(figsize=(10,15))
    for j in range(3):
        plt.subplot(3,1,j+1)
        test_i = random.randint(0, len(df))
        test_data = df.iloc[test_i]['file']
        data = load_mat(os.path.join(data_path, f'{test_data}.h5'))
        test_label = df.iloc[test_i]['label']

        for i in range(30):
            plt.plot(data['amp'][i,:])
        plt.title(f'{test_i}, {test_data}, {test_label}, {data["label"]}')
    plt.show()



if __name__ == '__main__':
    dataset_path = 'D:\study\dataset\wifi-partition-data-abs'
    dataset_path = os.path.join(dataset_path, 'wifi_partition_data_abs')

    save_path = 'dataset'
    save_path = os.path.join(save_path)

    # split_train_test(dataset_path, save_path, train_ratio=0.98)
    # print('='*20)
    # read_file_list('test_list.csv', dataset_path)
    # read_file_list('train_list.csv')

    check_data(os.path.join('dataset','test_list.csv'), os.path.join('dataset/test'))
