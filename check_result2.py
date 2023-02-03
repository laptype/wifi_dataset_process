
import os
import pandas as pd
import scipy.io as scio
import matplotlib.pyplot as plt
from PIL import Image

from util import log_f_ch

class Check_result():
    def __init__(self,
                 mode = 'test',
                 dataset_name = 'WiVio',
                 strategy_name = 'vit_span_cls_raw',
                 tab = 'day_1_8',
                 backbone_name = 'vit_b_16_0.2',
                 head_name = 'wifi_ar_span_cls',
                 check_point_path = os.path.join('/home/lanbo/wifi_wavelet/result/checkpoint/'),
                 dataset_path = os.path.join('/home/lanbo/dataset/wifi_violence/'),
                 save_path = os.path.join('/home/lanbo/dataset/wifi_dataset_process/','result')):

        self.save_path = os.path.join(save_path, backbone_name)

        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.strategy_name = strategy_name
        self.tab = tab
        self.backbone_name = backbone_name
        self.head_name = head_name
        self.check_point_path = os.path.join(
            check_point_path, '%s' % dataset_name, '%s' % strategy_name, '%s' % tab, '%s' % backbone_name
        )

        if mode == 'test':
            self.result_path = os.path.join(self.check_point_path, 'Test_dataset')
            self.Dataset_csv_path = os.path.join(self.result_path, 'test_dataset.csv')
            self.save_path = os.path.join(self.save_path, 'Test_dataset')
        else:
            self.result_path = os.path.join(self.check_point_path, 'Train_dataset')
            self.Dataset_csv_path = os.path.join(self.result_path, 'train_dataset.csv')
            self.save_path = os.path.join(self.save_path, 'Train_dataset')

        self.confusion_matrix_path = os.path.join(self.result_path, '%s-%s-%s-confusion_matrix.csv' % (
            backbone_name,
            head_name,
            'label',
        ))
        self.fileid_matrix_path = os.path.join(self.result_path, '%s-%s-%s-file_name.csv' % (
            backbone_name,
            head_name,
            'label',
        ))

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def read_file_name_csv(self):
        df = pd.read_csv(self.Dataset_csv_path, header=None)
        return df.values.tolist()

    def get_file_path(self, file_id):
        if file_id == '':
            return None

        file_id = int(file_id)
        # file_path = os.path.join(self.dataset_path, f'{file_name}.mat')
        # self.Dataset_csv_path
        file_list = self.read_file_name_csv()
        index, file_name = file_list[file_id]
        assert index == file_id, 'error in index'
        return file_name

    def load_data(self, file_name):
        file_path = os.path.join(self.dataset_path, f'{file_name}.mat')
        return scio.loadmat(file_path)['absMat']

    def draw_data(self, file_id_list, pred, gt):
        plt.figure(figsize=(30, 15))
        for i, id in enumerate(file_id_list):
            file_name = self.get_file_path(id)
            if file_name is not None:
                data = self.load_data(file_name)
                plt.subplot(4, 4, i + 1)
                plt.plot(data[:, :, 0].T)
                plt.title(f'{file_name}.mat')
            else:
                break
        plt.suptitle(f'pred = {pred} ground truth = {gt}')
        f = plt.gcf()  # 获取当前图像
        f.savefig(os.path.join(self.save_path, f'pred_{pred}_gt_{gt}.png'))
        f.clear()  # 释放内存
        plt.close()



    def generate_pic(self, pred, gt):

        df = pd.read_csv(self.fileid_matrix_path, header=None)
        file_id_list = df.values.tolist()

        # i = file_id_list[0][0][1:-2].split(', ')[5]
        # self.get_file_path(i)

        file_id = file_id_list[pred][gt][1:-1].split(', ')
        # print(file_id)
        for id in file_id:
            file_name = self.get_file_path(id)
            # print(file_name)

            # TODO: 画图
            """
                遍历每一个文件，然后找到相同的场景，人，动作
            """

    def show_pic(self, pred, gt):
        img = Image.open(os.path.join(self.save_path, f'pred_{pred}_gt_{gt}.png'))
        plt.figure(f'pred_{pred}_gt_{gt}.png', figsize=(30,15))

        plt.axis('off')
        plt.imshow(img)
        plt.show()

    def confusion_matrix(self):
        df = pd.read_csv(self.confusion_matrix_path, header=None)
        # print(df)
        df_list = df.values.tolist()
        correct_n = 0
        false_n = 0
        false_cls = [[] for _ in range(7)]
        for pred in range(7):
            for gt in range(7):
                if pred == gt:
                    correct_n += df_list[pred][gt]
                    false_cls[pred].append(df_list[pred][gt])
                else:
                    false_n += df_list[pred][gt]
                    false_cls[pred].append(df_list[pred][gt])

        print(f'正 {correct_n} 错 {false_n} 总 {correct_n+false_n} ' + '正确率 %2.3f' % (correct_n / (correct_n+false_n) * 100))

        print(log_f_ch('cls','正确','1st','2nd','3rd','4th','5th', '6th'))
        for index, cls in enumerate(false_cls):
            print(log_f_ch(str(index)), end='')
            for i in range(7):
                cls_1 = cls
                cls_1 = sorted(cls_1, reverse=True)
                max_index = cls.index(cls_1[0])
                print(log_f_ch(f'{max_index} ({cls_1[0]})'),end='')
                cls[max_index] = -1
            print('')
            # print(f'cls: {index}, {cls.index(cls_1[0])} ({cls_1[0]})')

def read_fileid_csv(csv_path):
    df = pd.read_csv(csv_path)

    print(df)


if __name__ == '__main__':

    """
        /home/lanbo/wifi_wavelet/result/checkpoint/WiVio/vit_span_cls_raw/day_1_8/vit_b_16_0.2/
    """
    ################################################################################################
    backbone_name_list = [
        'resnet1d_101',
        'resnet1d_50',
        'resnet1d_34',
        'resnet1d_18',
    ]
    for backbone_name in backbone_name_list:

        check_result = Check_result(mode='test',
                                    backbone_name = backbone_name,
                                    strategy_name = 'resnet1d_span_cls_raw_time',
                                    tab='day_1_12',
                                    dataset_path=os.path.join('/home/lanbo/dataset/wifi_violence_processed_loc/'))
        # check_result.generate_pic()
        # check_result.show_pic(1,5)
        print(backbone_name.center(90,'='))
        check_result.confusion_matrix()
        check_result.generate_pic(3,5)
        print(' '.center(90,' '))

    ################################################################################################

