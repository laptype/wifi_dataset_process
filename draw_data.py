import os
import random

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt


class Check_dataset():
    def __init__(self, data_path: os.path, save_path):
        self.data_path = data_path
        self.save_path = save_path

        self.n_class = 7
        self.n_location = 8


    def load_file(self):

        data_list = [[{} for _ in range(self.n_location)] for _ in range(self.n_class)]

        file_list = os.listdir(self.data_path)
        for i, file_name in enumerate(file_list):
            file_name, _ = file_name.split('.')
            data_info, num = file_name.split('_')
            location, person, action = data_info.split('-')

            if person not in data_list[int(action) - 1][int(location) - 1].keys():
                data_list[int(action) - 1][int(location) - 1][person] = []

            data_list[int(action) - 1][int(location) - 1][person].append(file_name)

        self.person_list = data_list[0][0].keys()
        return data_list

    def load_data(self, file_name):
        file_path = os.path.join(self.data_path, f'{file_name}.mat')
        data = scio.loadmat(file_path)['absMat']
        return data

    def plot_loc(self):

        data_list = self.load_file()
        index = random.randint(0, 10)

        for person in self.person_list:
            save_path = self.get_path(self.save_path, f'person {person}')
            save_path = self.get_path(save_path, f'same_cls')
            for cls in range(self.n_class):
                plt.figure(figsize=(15, 15))
                for loc in range(self.n_location):
                    data_name = data_list[cls][loc][person][index]
                    data = self.load_data(data_name)
                    plt.subplot(4,2,loc+1)
                    plt.plot(data[:,:,0].T)
                    plt.title(f'{data_name}.mat')
                plt.suptitle(f'person: {person}, class: {cls}')
                f = plt.gcf()  # 获取当前图像
                f.savefig(os.path.join(save_path, f'person-{person}_cls-{cls}.png'))
                f.clear()  # 释放内存
                plt.close()
                    # pass
    def get_path(self, path1, path2):
        save_path = os.path.join(path1, path2)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return save_path


if __name__ == '__main__':
    """
        路径相关
    """
    data_path = os.path.join('wifi_partition_data_abs')
    save_path = os.path.join('check_data')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # data_list = load_data(data_path)
    #
    # for i_cls, cls in enumerate(data_list):
    #     print(str(i_cls).center(20, '='))
    #     for i_loc, loc in enumerate(cls):
    #         print(str(i_loc).center(10,'-'))
    #         for k, v in loc.items():
    #             print(k, v)

    check = Check_dataset(data_path, save_path)
    data_list = check.load_file()
    # data = check.load_data(data_list[0][0]['12'][0])
    # plt.plot(data[:,:,0].T)
    # plt.show()
    check.plot_loc()
