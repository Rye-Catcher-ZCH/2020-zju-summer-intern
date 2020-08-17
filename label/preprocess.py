# 视频帧率25fps
import argparse
import cv2
import h5py
import numpy as np
from hog_feature import HogDescriptor
import os
a = []
b = []

# 提取视频帧,并全部标注为0
def label_data(args):
    dataroot = args.video_path
    datafile = h5py.File(args.data_store_path, 'w')
    crop_size = int(args.crop_size/2)
    sample = []
    label = []
    cap = cv2.VideoCapture(dataroot)
    k = 0
    i = 1
    while True:
        k = k + 1
        print(k)
        ret, frame = cap.read()
        if ret:  # 视频未读取完,有帧存在
            temp = frame[b[-1] - crop_size:b[-1] + crop_size, a[-1] - crop_size:a[-1] + crop_size, :]
            temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)
            temp = np.array(temp)
            # cv2.imshow('label',temp)
            frame = cv2.rectangle(frame, (a[-1] - crop_size, b[-1] - crop_size), (a[-1] + crop_size, b[-1] + crop_size), (0, 0, 255), 2)
            cv2.imshow('img', frame)
            cv2.waitKey(10)
            label_class = 0  #  先全部标注为0，然后再修改其中部分为进球帧
            print("No.{} frame, Label = {}".format(k, label_class))
            label.append(label_class)
            sample.append(temp)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    datafile.create_dataset('data', data=sample)
    datafile.create_dataset('label', data=label)
    datafile.close()

def on_Mouse(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        print(a[-1], b[-1])

# 将标注的txt文件写入data.h5
def annotation2dataset(args):
    annotation_path = args.annotation_path
    root_path = args.data_store_path
    store_path = args.anno_data_store_path
    file = open(annotation_path, 'r', encoding='utf-8')
    old_dataset = h5py.File(root_path, 'r')
    print([key for key in old_dataset.keys()])
    origin_data = old_dataset["data"][:]
    origin_label = old_dataset["label"][:]
    old_dataset.close()
    new_dataset = h5py.File(store_path, 'w')
    for line in file.readlines():  # 读取标注信息
        line = line.split()
        start_frame = line[0]
        end_frame = line[1]
        if start_frame == end_frame:  # 仅有一帧
            origin_label[int(start_frame)] = 1
        else:  # 有球帧标注为1
            for i in range(int(start_frame), int(end_frame) + 1):
                origin_label[i] = 1
    print(origin_label[:])
    new_dataset.create_dataset('data', data=origin_data)
    new_dataset.create_dataset('label', data=origin_label)
    new_dataset.close()
    file.close()

# 提取hog特征
def hog_extract(args):
    dataroot = args.anno_data_store_path
    storeroot = args.hog_feature_store_path
    hog = HogDescriptor(args.hog_cell_size, args.hog_block_size, args.hog_stride, args.hog_bins)
    feature = []
    temp_label = []
    datafile = h5py.File(dataroot, 'r')
    print(len(datafile['data']), len(datafile['label']))
    data = np.array(datafile['data'][0:])
    label = np.array(datafile['label'][0:])
    print(len(data), len(label))

    for i in range(100):
        feature.append(hog.calculate_hog(data[i])[0])
        # print(len(hog.calculate_hog(data[i])))
        temp_label.append(label[i])
        print('the %d picture has been prepared'%(i+1))
    storefile = h5py.File(storeroot, 'w')
    storefile.create_dataset('data', data=feature)
    storefile.create_dataset('label', data=temp_label)
    storefile.close()
    print('%d features and %d labels have been stored'%(len(feature),len(temp_label)))

# 2-frame hog
def make2frame(args):
    f = h5py.File(args.hog_feature_store_path, 'r')
    data = np.array(f['data'])
    label = np.array(f['label'])
    new_data = []
    new_label = []
    for i in range(len(label)-1):
        new_data.append(np.append(data[i], data[i+1]))
        new_label.append(label[i] or label[i+1])
    new_f = h5py.File("/Users/maitianshouwangzhe/Desktop/zju-2020-summer-intern/label/data/data3_hog_2frame.h5",'w')
    new_f.create_dataset('data', data=new_data)
    new_f.create_dataset('label', data=new_label)
    new_f.close()
    print("successfullt generate two frame")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 任务类型
    parser.add_argument('--task', type=str, default="label")
    # 视频路径
    parser.add_argument('--video_path', type=str, default="/Users/maitianshouwangzhe/Desktop/zju-2020-summer-intern/label/video/basketball-video-03.avi")
    # 提取视频帧存储路径(label全为0)
    parser.add_argument('--data_store_path', type=str, default="/Users/maitianshouwangzhe/Desktop/zju-2020-summer-intern/label/data/data3.h5")
    # 标注文件路径(.txt)
    parser.add_argument('--annotation_path', type=str, default="/Users/maitianshouwangzhe/Desktop/zju-2020-summer-intern/label/annotation.txt")
    # 标注后视频帧存储路径
    parser.add_argument('--anno_data_store_path', type=str, default="/Users/maitianshouwangzhe/Desktop/zju-2020-summer-intern/label/data/data3_anno.h5")
    # 提取hog特征后数据集存储路径
    parser.add_argument('--hog_feature_store_path', type=str, default="/Users/maitianshouwangzhe/Desktop/zju-2020-summer-intern/label/data/data3_hog.h5")
    # 球框帧大小
    parser.add_argument('--crop_size', type=int, default=100)
    # 特征类别,单帧hog或双帧hog
    parser.add_argument('--feature_type', type=str, default="one_frame_hog")
    # hog cell大小
    parser.add_argument('--hog_cell_size', type=int, default=8)
    # hog block大小
    parser.add_argument('--hog_block_size', type=int, default=2)
    # hog block滑动步长
    parser.add_argument('--hog_stride', type=int, default=8)
    # hog 角度数量
    parser.add_argument('--hog_bins', type=int, default=9)
    args = parser.parse_args()

    if args.task == "label":
        cap = cv2.VideoCapture(args.video_path)
        ret, frame = cap.read()
        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', on_Mouse)
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
        cap.release()
        cv2.destroyAllWindows()
        print(a[-1], b[-1])  # 打印方框坐标
        label_data(args)
    elif args.task == "annotation":
        annotation2dataset(args)
    elif args.task == "hog_extract":
        hog_extract(args)
        if args.feature_type == "two_frame_hog":
            make2frame(args)