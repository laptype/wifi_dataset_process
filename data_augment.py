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

class Data_augment():
    def __init__(self, datasource_path, save_path):
        self.datasource_path = datasource_path
        self.save_path = save_path

        self.train_data_path = os.path.join(datasource_path, 'train')
        self.train_list_path = os.path.join(datasource_path, 'train_list.csv')
        self.test_data_path = os.path.join(datasource_path, 'test')
        self.test_list_path = os.path.join(datasource_path, 'test_list.csv')



    def augment(self):
        pass

    def file_list(self):
        # data_list = pd.read_csv(self.train_list_path)
        data_list = pd.read_csv(self.test_list_path)
        data_list = list(data_list['file'])

        # data_list_loc = [[[] for _ ] for _ in range(8)]

        for file in data_list:
            data_info, num = file.split('_')
            location, person, action = data_info.split('-')

class Mean_augment(Data_augment):
    def __init__(self, datasource_path, save_path):
        super(Mean_augment, self).__init__(datasource_path, save_path)

    def augment(self):
        pass

if __name__ == '__main__':
    data_augment = Mean_augment('/home/lanbo/dataset/wifi_violence_processed_loc/', '')
    data_augment.file_list()
