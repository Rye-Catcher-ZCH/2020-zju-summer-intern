# coding=utf-8
import cv2
import h5py
import numpy as np

# 打印某一帧及其后续一系列帧
def show_frame(f_no, f_count, dataset_path):
    """
    :param f_no: 帧号
    :param f_count: 打印f_no号帧后续f_count帧
    """
    i_count = 0
    f = h5py.File(dataset_path, 'r')
    for i_count in range(0, f_count):
        image = f['data'][f_no + i_count]
        cv2.imshow('img', image)
        cv2.waitKey(500)
    return

# f = h5py.File('path/filename.h5','r') #打开h5文件
f = h5py.File('datasets/data3_hog.h5', 'r')

# 查看所有的主键
# print([key for key in f.keys()])
# 查看图片数据
# print(f['data'][:])
# print(f['data'][:].shape)
# print(f['label'][:].shape)
# show_frame(204, 15, 'data/data3.h5')
# show_frame(204, 15, 'data/data3_anno.h5')
