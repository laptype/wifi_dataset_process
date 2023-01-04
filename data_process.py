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


def split_train_test(dataset_path: os.path, save_path, rand = True, train_ratio = 0.8, mean_std_path=None):
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

    train_list_path = os.path.join(save_path, 'train_list.csv')
    test_list_path = os.path.join(save_path, 'test_list.csv')

    df_train.to_csv(train_list_path, index=False)
    df_test.to_csv(test_list_path, index=False)

    train_path = os.path.join(save_path, 'train')
    test_path = os.path.join(save_path, 'test')


    save_data_h5(test_list, test_path, dataset_path, mean_std_path)
    save_data_h5(train_list, train_path, dataset_path, mean_std_path)

    return train_path, train_list_path, test_path, test_list_path


def save_data_h5(file_list, save_path, dataset_path, mean_std_path = None):

    def _normalize(data, mean, std):
        return (data - mean) / std

    if mean_std_path is not None:
        mean_std = load_mat(mean_std_path)

    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    os.makedirs(save_path)

    for file, label in tqdm(file_list):

        data = scio.loadmat(os.path.join(dataset_path, file))['absMat']
        C, D, N = data.shape
        data = np.transpose(data, (2, 0, 1)).reshape((C * N, D))

        if mean_std_path is not None:
            data = _normalize(data, mean_std['mean'], mean_std['std'])

        save_data = {'amp': data,
                     'label': int(label)}

        file_path = os.path.join(save_path, f'{file}.h5')

        save_mat(file_path, save_data)


def _normalize(list_path, data_path):

    def get_mean_std(data):
        channel = data.shape[1]
        mean = np.mean(data.transpose(1, 0, 2).reshape(channel, -1), axis=-1)
        mean = np.expand_dims(mean, 1)
        std = np.std(data.transpose(1, 0, 2).reshape(channel, -1), axis=-1)
        std = np.expand_dims(std, 1)
        return mean, std

    def normalize(data, mean, std):
        return (data - mean) / std

    amp_mean, amp_std = get_mean_std(read_all(list_path, data_path))
    return amp_mean, amp_std


def normalize_data(train_list_path, test_list_path, train_data_path, test_data_path, save_path):

    def normalize(data_list_path, data_path):
        data_list = pd.read_csv(data_list_path)
        for index, row in tqdm(data_list.iterrows()):
            data = load_mat(os.path.join(data_path, f'{row["file"]}.h5'))
            data['amp'] = (data['amp'] - amp_mean) / amp_std
            save_mat(os.path.join(data_path, f'{row["file"]}.h5'), data)

    """
        计算训练数据集的均值方差代替全部数据集的均值方差
    """
    amp_mean, amp_std = _normalize(train_list_path, train_data_path)

    save_mat(os.path.join(save_path, f'mean_std.h5'),
             {
                 'mean': amp_mean,
                 'std': amp_std
             })

    normalize(train_list_path, train_data_path)
    normalize(test_list_path, test_data_path)

def normalize_data_h5(train_list_path, test_list_path, train_data_path, test_data_path, mean_std_path='dataset/mean_std_train.h5'):

    mean_std = load_mat(mean_std_path)

    def normalize(data_list_path, data_path):
        data_list = pd.read_csv(data_list_path)
        for index, row in tqdm(data_list.iterrows()):
            data = load_mat(os.path.join(data_path, f'{row["file"]}.h5'))
            data['amp'] = (data['amp'] - mean_std['mean']) / mean_std['std']
            save_mat(os.path.join(data_path, f'{row["file"]}.h5'), data)

    normalize(train_list_path, train_data_path)
    normalize(test_list_path, test_data_path)

def downsample_data(train_list_path, test_list_path, train_data_path, test_data_path, downsample_factor = 2):

    def _downsample(data_list_path, data_path):
        data_list = pd.read_csv(data_list_path)
        for index, row in tqdm(data_list.iterrows()):
            data = load_mat(os.path.join(data_path, f'{row["file"]}.h5'))
            n_channel, seq_len = data['amp'].shape
            data['amp'] = torch.unsqueeze(torch.Tensor(data['amp']), dim=0)
            data['amp'] = F.interpolate(data['amp'], seq_len // downsample_factor, mode='linear')
            data['amp'] = torch.squeeze(data['amp']).numpy()
            save_mat(os.path.join(data_path, f'{row["file"]}.h5'), data)
    _downsample(train_list_path, train_data_path)
    _downsample(test_list_path, test_data_path)

def check_data(list_path, data_path, index_list, name, save_path):
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
        # test_i = random.randint(0, len(df))
        test_i = index_list[j]
        test_data = df.iloc[test_i]['file']
        data = load_mat(os.path.join(data_path, f'{test_data}.h5'))
        test_label = df.iloc[test_i]['label']

        for i in range(30):
            plt.plot(data['amp'][i,:])
        plt.title(f'{test_i}, {test_data}, {test_label}, {data["label"]}')

    f = plt.gcf()  # 获取当前图像
    f.savefig(os.path.join(save_path,f'{name}.png'))

    plt.show()
    f.clear()  # 释放内存


def read_all(list_path, data_path):
    data_list = pd.read_csv(list_path)
    amp_list = []
    # label_list = []
    for index, row in data_list.iterrows():
        data = load_mat(os.path.join(data_path, f'{row["file"]}.h5'))
        amp_list.append(np.expand_dims(data['amp'], 0))
        # label_list.append(data['label'])
    return np.concatenate(amp_list, axis=0)


if __name__ == '__main__':
    dataset_path = 'wifi_partition_data_abs'
    dataset_path = os.path.join(dataset_path)

    save_path = 'dataset'
    save_path = os.path.join(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    '''
        数据集划分
    '''
    # train_path, train_list_path, test_path, test_list_path = split_train_test(dataset_path,
    #                                                                           save_path,
    #                                                                           train_ratio=0.1,
    #                                                                           mean_std_path='dataset/mean_std_train.h5')
    train_path, train_list_path, test_path, test_list_path = split_train_test(dataset_path,
                                                                              save_path,
                                                                              train_ratio=0.8,
                                                                              mean_std_path=None)
    # split_train_test(dataset_path, save_path, train_ratio=0.98)
    '''
        check_data
    '''
    # index_list = [5, 1000, 2000]
    index_list = [1,3,5]

    check_data(os.path.join('dataset','test_list.csv'), os.path.join('dataset/test'), index_list,'amp',save_path)
    # print(read_all(os.path.join('dataset/test_list.csv'), os.path.join('dataset/test')).shape)
    # normalize_data(train_list_path=train_list_path,
    #                test_list_path =test_list_path,
    #                train_data_path=train_path,
    #                test_data_path =test_path,
    #                save_path=save_path)
    # check_data(os.path.join('dataset','test_list.csv'), os.path.join('dataset/test'))
    '''
        归一化
    '''
    normalize_data_h5(train_list_path=train_list_path,
                      test_list_path =test_list_path,
                      train_data_path=train_path,
                      test_data_path =test_path,
                      mean_std_path = 'dataset/mean_std_train.h5')
    # mean_std = load_mat('dataset/mean_std_train.h5')
    # print(mean_std['mean'].shape, mean_std['std'].shape)
    check_data(os.path.join('dataset', 'test_list.csv'), os.path.join('dataset/test'),index_list,'amp_nor',save_path)

    '''
        下采样
    '''
    downsample_data(train_list_path=train_list_path,
                   test_list_path =test_list_path,
                   train_data_path=train_path,
                   test_data_path =test_path,
                   downsample_factor=2)

    check_data(os.path.join('dataset', 'test_list.csv'), os.path.join('dataset/test'),index_list,'amp_nor_down',save_path)