"""
38 x 68
1. 预处理: 降采样 192x168 -> 48x42,  一维向量
2. 分割数据集: 每类取 p 作为训练样本
3. 实验
   1. 5 fold 参数选择 稀疏: p=7, 13, 20  字典大小: 200, 400, 600
"""
import glob

from PIL import Image

import numpy as np
from sklearn.model_selection import RepeatedKFold, train_test_split


class Data(object):
    def __init__(self, fold=5, p=7, n_repeats=1):
        self.fold = fold
        self.p = p  # 7, 13, 70
        self.n_repeats = n_repeats
        self.data = dict()
        self.labels = np.zeros(2470, dtype=np.int16)
        self.features = np.zeros((2470, 2016))
        self.test_x_ = None
        self.test_y_ = None

    def preprocess(self):
        """
        预处理: 降采样 192x168 -> 48x42,  一维向量
        """
        face_pathes = glob.glob(r"data/*")
        face_names = [path.split("/")[1] for path in face_pathes]

        index = 0
        for face_name in face_names:
            faces = glob.glob("data/%s/*.pgm" % face_name)
            face_class = face_name[-2:]
            self.labels[index] = int(face_class)
            for face in faces:
                image = Image.open(face)
                image = image.resize((48, 42))
                image = np.asarray(image)
                self.features[index, :] = image.flatten()
                index += 1

        assert index == 2470

    def get_val_test_data(self):
        train_x, self.test_x_, train_y, self.test_y_ = train_test_split(
            self.features, self.labels, train_size=self.p * 38, shuffle=True
        )
        # kf = KFold(n_splits=self.fold)
        kf = RepeatedKFold(n_splits=self.fold, n_repeats=self.n_repeats)
        for val_index, test_index in kf.split(train_x):
            train_x, val_x = train_x[val_index], train_x[test_index]
            train_y, val_y = train_y[val_index], train_y[test_index]
            yield train_x, val_x, train_y, val_y

    def get_test_data(self):
        return self.test_x_, self.test_y_

    def get_data(self):
        return self.features, self.labels
