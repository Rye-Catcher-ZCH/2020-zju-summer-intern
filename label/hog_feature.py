import math
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np


class HogDescriptor():
    def __init__(self, cell_size=8, block_size=2, stride=8, bins=9):
        self.cell_size = cell_size
        self.block_size = block_size
        self.stride = stride
        self.bins = bins

    # def extract(self):
    def calculate_hog(self, img):
        """
        :param img: 原始图片
        :return: 原始图片的hog特征和hog特征可视化图
        """
        # preprocessing
        # img = self.padding(img)
        img = np.sqrt(img / np.max(img))
        img = img * 255

        _, img_size = img.shape
        grad_mag, grad_ang = self.calculate_pixel_gradient(img)  # 计算图片中的每个像素点的梯度和角度
        grad_mag = abs(grad_mag)  # 梯度幅值取绝对值

        # 将整张图片的梯度幅值和角度按照每个cell进行划分,计算每个cell的hog特征并存储到cell_hog_feature中
        cell_hog_feature = np.zeros(
            (int(img_size / self.cell_size), int(img_size / self.cell_size), self.bins))  # 每个元素是一个9维向量
        for i in range(cell_hog_feature.shape[0]):
            for j in range(cell_hog_feature.shape[1]):
                cell_mag = grad_mag[i * self.cell_size:(i + 1) * self.cell_size,
                           j * self.cell_size:(j + 1) * self.cell_size]
                cell_ang = grad_ang[i * self.cell_size:(i + 1) * self.cell_size,
                           j * self.cell_size:(j + 1) * self.cell_size]
                cell_hog_feature[i][j] = self.calculate_cell_hog(cell_mag, cell_ang)

        hog_image = self.generate_hog_image(np.zeros([img_size, img_size]), cell_hog_feature)  # 产生hog特征图
        hog_feature = []

        if self.block_size == 1:
            for i in range(cell_hog_feature.shape[0]):
                for j in range(cell_hog_feature.shape[1]):
                    block_hog_feature = []  # 计算block的hog特征,一个block 4个cell
                    block_hog_feature.extend(cell_hog_feature[i][j])
                    mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))  # 计算block向量的模长(二范数,用于归一化)
                    magnitude = mag(block_hog_feature)
                    if magnitude != 0:
                        normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                        block_hog_feature = normalize(block_hog_feature, magnitude)
                    hog_feature.append(block_hog_feature)  # 图片的hog特征为每个block特征的concat
        elif self.block_size == 2:
            for i in range(cell_hog_feature.shape[0] - 1):
                for j in range(cell_hog_feature.shape[1] - 1):
                    block_hog_feature = []  # 计算block的hog特征,一个block 4个cell
                    block_hog_feature.extend(cell_hog_feature[i][j])
                    block_hog_feature.extend(cell_hog_feature[i][j + 1])
                    block_hog_feature.extend(cell_hog_feature[i + 1][j])
                    block_hog_feature.extend(cell_hog_feature[i + 1][j + 1])
                    mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))  # 计算block向量的模长(二范数,用于归一化)
                    magnitude = mag(block_hog_feature)
                    if magnitude != 0:
                        normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                        block_hog_feature = normalize(block_hog_feature, magnitude)
                    hog_feature.append(block_hog_feature)  # 图片的hog特征为每个block特征的concat
        elif self.block_size == 3:
            for i in range(cell_hog_feature.shape[0] - 2):
                for j in range(cell_hog_feature.shape[1] - 2):
                    block_hog_feature = []  # 计算block的hog特征,一个block 4个cell
                    block_hog_feature.extend(cell_hog_feature[i][j])
                    block_hog_feature.extend(cell_hog_feature[i][j + 1])
                    block_hog_feature.extend(cell_hog_feature[i][j + 2])
                    block_hog_feature.extend(cell_hog_feature[i + 1][j])
                    block_hog_feature.extend(cell_hog_feature[i + 1][j + 1])
                    block_hog_feature.extend(cell_hog_feature[i + 1][j + 2])
                    block_hog_feature.extend(cell_hog_feature[i + 2][j])
                    block_hog_feature.extend(cell_hog_feature[i + 2][j + 1])
                    block_hog_feature.extend(cell_hog_feature[i + 2][j + 2])
                    mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))  # 计算block向量的模长(二范数,用于归一化)
                    magnitude = mag(block_hog_feature)
                    if magnitude != 0:
                        normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                        block_hog_feature = normalize(block_hog_feature, magnitude)
                    hog_feature.append(block_hog_feature)  # 图片的hog特征为每个block特征的concat
        hog_feature = np.array(hog_feature, dtype=float)
        hog_feature = hog_feature.flatten()
        return hog_feature, hog_image

    def calculate_pixel_gradient(self, img):
        """
        计算图片每个像素的梯度幅值和角度
        :param img: 原始图片
        :return: 每个像素点的梯度幅值和角度
        """
        gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        grad_mag = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        grad_ang = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return grad_mag, grad_ang

    def calculate_cell_hog(self, cell_magnitude, cell_angle):
        """
        计算一个cell中的hog特征
        :param cell_magnitude:一个cell中所有像素点梯度的幅值
        :param cell_angle:一个cell中所有像素点梯度的角度
        :return:这个cell的hog特征
        """
        cell_hog_feature = [0] * self.bins  # 存储cell的hog特征

        interval = int(360/self.bins)  # 角度间隔
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                # 对cell中每一个像素,将其幅值和角度划分到对应的9维向量中,按照比例划分
                mag = cell_magnitude[i][j]
                ang = cell_angle[i][j]

                left = int(ang / interval)  # 左区间序号
                if ang == 360:  # 考虑360度的特殊情况
                    left = 8
                    right = 0
                if left == self.bins - 1:  # 如果左区间是最后一个
                    right = 0  # 右区间取0
                else:
                    right = left + 1
                # 不仅仅在一个区间增加计数,而是在左右两个区间按照比例计数
                right_ratio = ang / interval - left
                left_ratio = 1 - right_ratio
                cell_hog_feature[left] += (mag * left_ratio)
                cell_hog_feature[right] += (mag * right_ratio)

        return cell_hog_feature

    def generate_hog_image(self, image, cell_hog_feature):
        """
        :param image: 用于存储hog特征可视化图
        :param cell_hog_feature: 原始图片每个cell的hog特征
        :return: 原始图片的hog特征可视化图
        """
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_hog_feature).max()
        for x in range(cell_hog_feature.shape[0]):
            for y in range(cell_hog_feature.shape[1]):
                cell_grad = cell_hog_feature[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = int(360/self.bins)
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image


# test
# f = h5py.File('datasets/data3_anno.h5', 'r')
# img = f['data'][1300]
# hog = HogDescriptor(block_size=3)
# vector, image = hog.calculate_hog(img)  # 提取hog特征和hog特征可视化图
# print(type(vector))
# print(np.array(vector).shape)
# plt.imshow(image, cmap=plt.cm.gray)  # hog特征可视化
# plt.show()
