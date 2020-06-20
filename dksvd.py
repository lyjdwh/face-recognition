import time

import numpy as np
import scipy as sp
import scipy.linalg as splin
from data import Data
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import orthogonal_mp_gram
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize


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
            wk = X[j, :] > 0
            if sp.sum(wk) == 0:
                continue

            D[:, j] = 0
            Ekr = Y[:, wk] - D.dot(X[:, wk])
            u, s, vt = splin.svd(Ekr)
            d = u[:, 0]
            d = normalize(d, axis=0)
            D[:, j] = d
            X[j, :] = sp.dot(vt.T[:, 0], s[0, 0])

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
        if n_nonzero_coefs is None:
            n_nonzero_coefs = int(0.1 * Y.shape[1])

        return orthogonal_mp_gram(
            gram, Xy, copy_Gram=False, copy_Xy=False, n_nonzero_coefs=n_nonzero_coefs
        )

    def _ksvd_fit(self, Y, Dinit=None):
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

        for i in range(self.n_iter):
            X = self._transform(D, Y)
            e = splin.norm(Y - D.dot(X))
            if e < self.tol:
                break
            D, X = self._update_dict(Y, D, X)

        return D, X

    def fit(self, training_feats, labels, Dinit=None):

        """
        Input
        ----------
        training_feats  -training features (shape = [n_samples, n_features])
        labels          -label matrix for training feature (numberred from 1 to nb of classes)
        Dinit           -initial guess for dictionary
        """
        training_feats = training_feats.T

        self.classes_, labels = np.unique(labels, return_inverse=True)

        H_train = sp.zeros((int(labels.max()), training_feats.shape[1]), dtype=float)
        for c in range(int(labels.max())):
            H_train[c, labels == (c + 1)] = 1.0

        W = np.concatenate((training_feats, H_train), axis=0)

        P, X = self._ksvd_fit(W, Dinit)
        self.D_ = P[: training_feats.shape[0], :]
        self.C_ = P[training_feats.shape[0] :, :]
        self.D_ = normalize(self.D_, axis=0)
        self.C_ = normalize(self.C_, axis=0)
        return self

    def predict(self, Y):
        """
        predict single data
        """
        Y = Y[np.newaxis, :].T
        X = self._transform(self.D_, Y)
        L = sp.dot(self.C_, X)
        return self.classes_[L.argmax(axis=0)][0]


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
    for i in range(len(dictsizes)):
        dictsize = dictsizes[i]
        p = ps[i]
        param_grid = {
            "dictsize": [dictsize],
            "n_iter": [50],
            "tol": [1e-6],
            "sparsitythres": np.linspace(10, 50, 9),
        }

        fivefolds = GridSearchCV(DKSVD(), param_grid, cv=5, verbose=1, n_jobs=10)
        data = Data(p=p)
        features, labels = data.get_data()

        fivefolds.fit(features, labels)

        print("The best estimator found by GridSearch:")
        print(fivefolds.best_estimator_)
