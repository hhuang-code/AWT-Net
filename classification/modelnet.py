# All functions defined in the ModelNet40 class are referred from https://github.com/mutianxu/GDANet/blob/main/util/data_util.py

import os
import sys
sys.path.insert(0, '..')

import glob
import h5py
import numpy as np

from torch.utils.data import Dataset

from data_utils import translate_pointcloud

import pdb


def load_data(data_path, mode):
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(data_path, 'ply_data_%s*.h5' % mode)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)     # already centered and normalized
    all_label = np.concatenate(all_label, axis=0)

    return all_data, all_label


def load_name(data_path):
    with open(os.path.join(data_path, 'shape_names.txt'), 'r') as f:
        name = f.readlines()
        name = [x.strip() for x in name]

    return name


class ModelNet40(Dataset):
    def __init__(self, data_path, n_pts, mode='train'):
        self.data_path = data_path
        self.n_pts = n_pts
        self.mode = mode

        self.data, self.label = load_data(self.data_path, self.mode)

        self.name = load_name(self.data_path)

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.n_pts]
        label = self.label[item]
        name = self.name[label.item()]
        if self.mode == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)  # shuffle points

        return pointcloud, label, name

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40('../data/modelnet40_ply_hdf5_2048', 1024, 'train')
    test = ModelNet40('../data/modelnet40_ply_hdf5_2048', 1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)