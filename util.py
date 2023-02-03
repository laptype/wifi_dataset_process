import os
import h5py
import numpy as np


def load_mat(path: os.path):
    result = {}
    with h5py.File(path, mode='r') as file:
        for key in file.keys():
            result[key] = np.array(file[key])
    return result


def save_mat(path: os.path, data):
    with h5py.File(path, mode='w') as file:
        for key, value in data.items():
            file[key] = value

def pad_len(string, length):
    return length - len(string.encode('GBK')) + len(string)

def log_f_ch(*str_list, str_len: list = None):

    if str_len is None:
        str_len = [10 for _ in range(len(str_list))]
    else:
        assert len(str_len) == len(str_list) ,"length of 'str_len' must equal to length of 'str_list'"

    output_str = ''
    for i, str in enumerate(str_list):
        output_str += "{0:<{len1}}\t".format(str, len1=pad_len(str, str_len[i]))
    return output_str