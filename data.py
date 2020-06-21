"""
38 x 65
1. 预处理: 降采样 192x168 -> 48x42,  一维向量
2. 分割数据集: 每类取 p 作为训练样本
3. 实验
   1. 5 fold 参数选择 稀疏: p=7, 13, 20  字典大小: 200, 400, 600
"""
import glob
from random import shuffle

from PIL import Image

import numpy as np
from sklearn.model_selection import RepeatedKFold, train_test_split


class Data(object):
    def __init__(self, fold=5, p=7, n_repeats=1):
        self.fold = fold
        self.p = p  # 7, 13, 70
        self.n_repeats = n_repeats
        self.train_x_ = list()
        self.train_y_ = list()
        self.test_x_ = list()
        self.test_y_ = list()

    def preprocess(self):
        """
        预处理: 降采样 192x168 -> 48x42,  一维向量
        """
        face_pathes = glob.glob(r"data/*")
        face_names = [path.split("/")[1] for path in face_pathes]

        for face_name in face_names:
            faces = glob.glob("data/%s/*.pgm" % face_name)

            shuffle(faces)
            face_class = int(face_name[-2:])
            index = 0
            for face in faces:
                image = Image.open(face)
                image = image.resize((48, 42))
                image = np.asarray(image).flatten()
                if index < self.p:
                    self.train_x_.append(image)
                    self.train_y_.append(face_class)
                else:
                    self.test_x_.append(image)
                    self.test_y_.append(face_class)
                index += 1
        self.train_x_ = np.asarray(self.train_x_)
        self.train_y_ = np.asarray(self.train_y_)

        # shuffle
        self.train_x_, _, self.train_y_, _ = train_test_split(
            self.train_x_, self.train_y_, test_size=1e-10, shuffle=True
        )

        self.test_x_ = np.asarray(self.test_x_)
        self.test_y_ = np.asarray(self.test_y_)
        # shuffle
        self.test_x_, _, self.test_y_, _ = train_test_split(
            self.test_x_, self.test_y_, test_size=1e-10, shuffle=True
        )

    def get_val_test_data(self):
        # kf = KFold(n_splits=self.fold)
        kf = RepeatedKFold(n_splits=self.fold, n_repeats=self.n_repeats)
        for val_index, test_index in kf.split(self.train_x_):
            train_x, val_x = self.train_x_[val_index], self.train_x_[test_index]
            train_y, val_y = self.train_y_[val_index], self.train_y_[test_index]
            yield train_x, val_x, train_y, val_y

    def get_test_data(self):
        return self.test_x_, self.test_y_

    def get_train_data(self):
        return self.train_x_, self.train_y_
