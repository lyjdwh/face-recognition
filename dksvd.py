import time

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.linalg as splin
from data import Data
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.externals import joblib
from sklearn.linear_model import orthogonal_mp_gram
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.preprocessing import normalize


def compute_accuracy(test_x, test_y, model):
    predict_y = model.predict(test_x)
    accuracy = sum(predict_y == test_y) / len(test_y)
    return accuracy


class DKSVD(BaseEstimator, ClassifierMixin):
    def __init__(self, dictsize=200, n_iter=10, tol=1e-6, sparsitythres=None):
        """
        Input
        ----------
        dictsize: Number of dictionary elements
        n_iter: Maximum number of iterations
        tol: tolerance for error
        sparsitythres: sparsity threshold
        """
        self.n_iter = n_iter
        self.tol = tol
        self.dictsize = dictsize
        self.sparsitythres = sparsitythres

    def _update_dict(self, Y, D, X):
        for j in range(self.dictsize):
            I = X[j, :] > 0
            if sp.sum(I) == 0:
                continue

            D[:, j] = 0
            g = X[j, I]
            r = Y[:, I] - D.dot(X[:, I])
            d = r.dot(g)
            d /= splin.norm(d)
            g = r.T.dot(d)
            D[:, j] = d
            X[j, I] = g.T
        return D, X

    def _initialize(self, Y):
        if min(Y.shape) < self.dictsize:
            D = sp.random.randn(Y.shape[0], self.dictsize)
        else:
            u, s, vt = sp.sparse.linalg.svds(Y, k=self.dictsize)
            D = sp.dot(u, sp.diag(s))
        D = normalize(D, axis=0)

        return D

    def _transform(self, D, Y):

        gram = D.T.dot(D)
        Xy = D.T.dot(Y)

        n_nonzero_coefs = self.sparsitythres

        return orthogonal_mp_gram(
            gram, Xy, copy_Gram=False, copy_Xy=False, n_nonzero_coefs=n_nonzero_coefs
        )

    def _ksvd_fit(self, Y, Dinit=None, islog=True):
        """
        Use data to learn dictionary and activations.
        Input
        ----------
        Y: data. (shape = [n_features, n_samples])
        Dinit: initialization of dictionary. (shape = [n_features, dictsize])
        Outputs
        ----------
        D: dictionary
        X: Sparse representation
        """
        if Dinit is None:
            D = self._initialize(Y)
        else:
            D = normalize(Dinit, axis=0)

        self.tes_ = list()
        self.ves_ = list()
        for i in range(self.n_iter):
            X = self._transform(D, Y)
            e = splin.norm(Y - D.dot(X))
            self.tes_.append(e)

            if islog:
                print("error:", e)
            if e < self.tol:
                break
            D, X = self._update_dict(Y, D, X)

        return D, X

    def fit(self, training_feats, labels, Dinit=None, islog=True):
        """
        Input
        ----------
        training_
        feats  -training features (shape = [n_samples, n_features])
        labels          -label matrix for training feature (numberred from 1 to nb of classes)
        Dinit           -initial guess for dictionary
        islog           -whether log
        """
        training_feats = training_feats.T

        self.classes_, labels = np.unique(labels, return_inverse=True)

        H_train = sp.zeros((38, training_feats.shape[1]), dtype=float)

        for c in range(38):
            H_train[c, labels == (c + 1)] = 1.0

        W = np.concatenate((training_feats, H_train), axis=0)

        P, X = self._ksvd_fit(W, Dinit, islog=islog)
        self.D_ = P[:-38, :]
        self.C_ = P[-38:, :]
        self.D_ = normalize(self.D_, axis=0)
        self.C_ = normalize(self.C_, axis=0)
        return self

    def predict(self, Y):
        """
        predict  data
        """
        Y = Y.T
        X = self._transform(self.D_, Y)
        L = sp.dot(self.C_, X)

        return self.classes_[L.argmax(axis=0)]


def cross_validation(dictsize, p):
    """
    another cross validation version
    """

    # data preprocess
    data = Data(p=p)
    data.preprocess()

    # 交叉验证求最优参数
    models = []
    test_accuracys = []
    sparsitythres = np.linspace(10, 50, 9).astype(np.int)
    for sparsitythre in sparsitythres:

        dksvd = DKSVD(dictsize=dictsize, n_iter=5, tol=200, sparsitythres=sparsitythre)

        val_accuracy = 0.0
        for train_x, val_x, train_y, val_y in data.get_val_test_data():
            dksvd.fit(train_x, train_y, islog=islog)
            val_accuracy += compute_accuracy(val_x, val_y, dksvd)

        val_accuracy = val_accuracy / 5
        print("val accuracy: ", val_accuracy)

        features, labels = data.get_train_data()
        dksvd.fit(features, labels, islog=islog)
        test_x, test_y = data.get_test_data()

        starttime = time.time()
        test_accuracy = compute_accuracy(test_x, test_y, dksvd)
        endtime = time.time()
        speed = (endtime - starttime) / len(test_y)

        print("test accuracy: ", test_accuracy)
        print("test speed %f s/image" % speed)

        models.append(dksvd)
        test_accuracys.append(test_accuracy)

    best_index = test_accuracys.index(max(test_accuracys))
    best_model = models[best_index]
    best_sparsitythre = sparsitythres[best_index]

    print("best sparsitythre: ", best_sparsitythre)


if __name__ == "__main__":
    """
    1. 5fold交叉验证 参数选择
    2. 训练流程
    3. 每组实验重复 10 次,训练结果(验证集和测试集平均精度和每幅图片平均时间)
    4. 保存模型
    5. 并行
    """

    dictsizes = [200, 400, 600]
    ps = [7, 13, 20]
    n_iter = 100
    tol = 1e-6
    islog = False
    n_jobs = 10
    for i in range(len(dictsizes)):
        dictsize = dictsizes[i]
        p = ps[i]

        print("dictsize: ", dictsize, "p:", p)

        # 交叉验证选择参数
        param_grid = {
            "dictsize": [dictsize],
            "n_iter": [n_iter],
            "tol": [tol],
            "sparsitythres": np.linspace(10, 50, 1).astype(np.int),
        }
        fivefolds = GridSearchCV(DKSVD(), param_grid, cv=5, verbose=1, n_jobs=n_jobs)
        data = Data(p=p)
        data.preprocess()
        features, labels = data.get_train_data()

        fivefolds.fit(features, labels, islog=islog)

        print("The best estimator found by GridSearch:")
        print(fivefolds.best_estimator_)

        skvd = DKSVD(
            dictsize=dictsize,
            n_iter=n_iter,
            tol=tol,
            sparsitythres=fivefolds.best_estimator_.sparsitythres,
        )

        skvd.fit(features, labels, islog=islog)
        train_errors = skvd.tes_
        x = list(range(1, n_iter + 1))

        plt.plot(x, train_errors, "r-x")
        plt.xlabel("training iterations")
        plt.ylabel("training error")
        plt.title("dksvd learning curve")
        plt.savefig("./results/dksvd-%d-%d" % (dictsize, p))
        plt.clf()

        # test
        test_x, test_y = data.get_test_data()

        starttime = time.time()
        test_accuracy = compute_accuracy(test_x, test_y, fivefolds.best_estimator_)
        endtime = time.time()
        speed = (endtime - starttime) / len(test_y)

        print("test accuracy: ", test_accuracy)
        print("test speed %f s/image" % speed)

        # save models
        joblib.dump(fivefolds.best_estimator_, "./models/dksvd-%d-%d" % (dictsize, p))
