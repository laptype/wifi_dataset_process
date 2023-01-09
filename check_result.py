
import os
import pandas as pd
import scipy.io as scio
import matplotlib.pyplot as plt
from PIL import Image

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



    def generate_pic(self):

        df = pd.read_csv(self.fileid_matrix_path, header=None)
        file_id_list = df.values.tolist()

        i = file_id_list[0][0][1:-2].split(', ')[5]
        self.get_file_path(i)

        for pred in range(7):
            for gt in range(7):

                file_id = file_id_list[pred][gt][1:-2].split(', ')
                file_id = file_id[:16]
                print(f'pred {pred} gt {gt}')
                self.draw_data(file_id, pred, gt)
                # print(len(file_id))

    def show_pic(self, pred, gt):
        img = Image.open(os.path.join(self.save_path, f'pred_{pred}_gt_{gt}.png'))
        plt.figure(f'pred_{pred}_gt_{gt}.png', figsize=(30,15))

        plt.axis('off')
        plt.imshow(img)
        plt.show()

    def confusion_matrix(self):
        df = pd.read_csv(self.confusion_matrix_path, header=None)
        print(df)


def read_fileid_csv(csv_path):
    df = pd.read_csv(csv_path)
    print(df)


if __name__ == '__main__':

    """
        /home/lanbo/wifi_wavelet/result/checkpoint/WiVio/vit_span_cls_raw/day_1_8/vit_b_16_0.2/
    """
    ################################################################################################

    check_result = Check_result(mode='test',
                                backbone_name = 'vit_b_16_0.2',
                                tab='day_1_8')
    # check_result.generate_pic()
    check_result.show_pic(1,5)
    check_result.confusion_matrix()

    ################################################################################################

