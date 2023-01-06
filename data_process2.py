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

def


if __name__ == '__main__':

    """
        路径相关：
    """
    dataset_path    = ''
    save_path       = ''
    train_path      = os.path.join(save_path, 'train')
    train_list_path = os.path.join(save_path, 'train_list.csv')
    test_path       = os.path.join(save_path, 'test')
    test_list_path  = os.path.join(save_path, 'test_list.csv')
    mean_std_path   = os.path.join(save_path, 'mean_std.h5')

    dataset_path = 'wifi_partition_data_abs'
    dataset_path = os.path.join(dataset_path)

    save_path = '/home/lanbo/dataset/wifi_violence_processed/'
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