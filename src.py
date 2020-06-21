import os
import time

import joblib
import numpy as np
import scipy as sp
import scipy.linalg as splin
from data import Data
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import orthogonal_mp_gram
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize


def compute_accuracy(test_x, test_y, model):
    predict_y = model.predict(test_x)
    accuracy = sum(predict_y == test_y) / len(test_y)
    return accuracy


class SRC(BaseEstimator, ClassifierMixin):
    def __init__(self, sparsitythres=None):
        self.sparsitythres = sparsitythres

    def _transform(self, D, Y):
        gram = D.T.dot(D)
        Xy = D.T.dot(Y)

        n_nonzero_coefs = self.sparsitythres

        return orthogonal_mp_gram(
            gram, Xy, copy_Gram=False, copy_Xy=False, n_nonzero_coefs=n_nonzero_coefs
        )

    def fit(self, training_feats, labels):
        training_feats = training_feats.T
        self.D_ = normalize(training_feats, axis=0)
        self.classes_ = labels

    def predict(self, Y):
        """
        predict  data
        """
        Y = Y.T

        predict_ys = []

        for i in range(Y.shape[1]):
            y = Y[:, i]  # print(y.shape)
            x = self._transform(self.D_, y)

            rs = []
            for j in range(x.shape[0]):
                e = splin.norm(y - self.D_[:, j] * x[j])
                rs.append(rs)

            predict_y = self.classes_[rs.index(min(rs))]
            predict_ys.append(predict_y)

        return np.array(predict_ys)


if __name__ == "__main__":
    """
    1. 5fold交叉验证 参数选择
    2. 训练流程
    3. 每组实验重复 10 次,训练结果(验证集和测试集平均精度和每幅图片平均时间)
    4. 保存模型
    5. 并行
    """

    ps = [7, 13, 20]
    n_jobs = 10  # Number of jobs to run in parallel.
    verbose = 0  # Controls the verbosity: the higher, the more messages
    for i in range(len(ps)):
        p = ps[i]

        print("p:", p)

        # 交叉验证选择参数
        param_grid = {
            "sparsitythres": np.linspace(10, 50, 1).astype(np.int),
        }
        fivefolds = GridSearchCV(SRC(), param_grid, cv=5, verbose=0, n_jobs=n_jobs)
        data = Data(p=p)
        data.preprocess()
        features, labels = data.get_train_data()

        fivefolds.fit(features, labels)

        print("The best estimator found by GridSearch:")
        print(fivefolds.best_estimator_)

        src = SRC(sparsitythres=fivefolds.best_estimator_.sparsitythres)

        src.fit(features, labels)

        # test
        test_x, test_y = data.get_test_data()

        starttime = time.time()
        test_accuracy = compute_accuracy(test_x, test_y, fivefolds.best_estimator_)
        endtime = time.time()
        speed = (endtime - starttime) / len(test_y)

        print("test accuracy: ", test_accuracy)
        print("test speed %f s/image" % speed)

        # save models
        if not os.path.exists("./models"):
            os.makedirs("./models")
        joblib.dump(fivefolds.best_estimator_, "./models/src-%d" % p)
