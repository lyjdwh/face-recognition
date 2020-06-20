# 38 x 68
# 1. 预处理: 降采样 192x168 -> 48x42,  一维向量
# 2. 分割数据集: 每类取 p 作为训练样本
# 3. 实验
#    1. 5 fold 参数选择 稀疏: p=7, 13, 20  字典大小: 200, 400, 600
# TODO 每个文件夹有一张特别大的图，名字和其他图片不一样
import glob

from PIL import Image

import numpy as np


class Data(object):
    def __init__(self, fold):
        self.fold = fold
        self.data = dict()

    def preprocess(self):
        """
        预处理: 降采样 192x168 -> 48x42,  一维向量
        """
        face_pathes = glob.glob(r"data/*")
        face_names = [path.split("/")[1] for path in face_pathes]

        for face_name in face_names:
            faces = glob.glob("data/%s/*.pgm" % face_name)
            face_class = face_name[-2:]
            self.data[face_class] = list()
            for face in faces:
                image = Image.open(face)
                image = image.resize((48, 42))
                image = np.asarray(image)
                self.data[face_class].append(image.flatten())

    def get_svm_data(self):
        pass

    def get_xs_data(self, p):
        pass
