import os
import sys
sys.path.insert(0, '..')

import h5py
import numpy as np

from torch.utils.data import Dataset

from data_utils import rotate_pointcloud, translate_pointcloud

import pdb


def load_h5(data_path, mode, with_bg):
    if mode == 'train':
        if with_bg:
            h5_filename = os.path.join(data_path, 'main_split', 'training_objectdataset.h5')
        else:
            h5_filename = os.path.join(data_path, 'main_split_nobg', 'training_objectdataset.h5')
    elif mode == 'test':
        if with_bg:
            h5_filename = os.path.join(data_path, 'main_split', 'test_objectdataset.h5')
        else:
            h5_filename = os.path.join(data_path, 'main_split_nobg', 'test_objectdataset.h5')
    else:
        raise ValueError('Mode {} is not supported.'.format(mode))

    f = h5py.File(h5_filename)
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')   # 15 classes
    f.close()

    return data, label


class ScanObjectNN(Dataset):
    def __init__(self, data_path, n_pts, mode='train', with_bg=True):
        self.data_path = data_path
        self.n_pts = n_pts
        self.mode = mode
        self.with_bg = with_bg

        self.data, self.label = load_h5(self.data_path, self.mode, with_bg)

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.n_pts]
        label = self.label[item]
        if self.mode == 'train':
            pointcloud = rotate_pointcloud(pointcloud[np.newaxis, ...]).squeeze(0)
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ScanObjectNN('../data/ScanObjectNN/h5_files', 1024, 'train', False)
    test = ScanObjectNN('../data/ScanObjectNN/h5_files', 1024, 'test', False)
    for data, label in train:
        print(data.shape)
        print(label.shape)