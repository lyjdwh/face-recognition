import numpy as np
import scipy as sp
import scipy.linalg as splin
from sklearn.linear_model import orthogonal_mp_gram


class DKSVD(object):
    def __init__(self, dictsize, max_iter=10, tol=1e-6, sparsitythres=None):
        """
        Input
        ----------
        dictsize: Number of dictionary elements
        max_iter: Maximum number of iterations
        tol: tolerance for error
        sparsitythres: sparsity threshold
        """
        self.D_ = None
        self.C_ = None
        self.max_iter = max_iter
        self.tol = tol
        self.dictsize = dictsize
        self.transform_n_nonzero_coefs = sparsitythres

    def _update_dict(self, Y, D, X):
        for j in range(self.n_components):
            wk = X[j, :] > 0
            if sp.sum(wk) == 0:
                continue

            D[:, j] = 0
            Ekr = Y[:, wk] - D.dot(X[:, wk])
            u, s, vt = splin.svd(Ekr)
            d = u[:, 0]
            d /= splin.norm(d)
            D[:, j] = d
            X[j, :] = sp.dot(vt.T[:, 0], s[0, 0])

        return D, X

    def _initialize(self, Y):
        if min(Y.shape) < self.n_components:
            D = sp.random.randn(Y.shape[0], self.n_components)
        else:
            u, s, vt = sp.sparse.linalg.svds(Y, k=self.n_components)
            D = sp.dot(u, sp.diag(s))
        D /= splin.norm(D, axis=0)[sp.newaxis, :]
        return D

    def _transform(self, D, Y):
        gram = D.T.dot(D)
        Xy = D.T.dot(Y)

        n_nonzero_coefs = self.transform_n_nonzero_coefs
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
        Dinit: initialization of dictionary. (shape = [n_features, n_components])
        Outputs
        ----------
        D: dictionary
        X: Sparse representation
        """
        if Dinit is None:
            D = self._initialize(Y)
        else:
            D = Dinit / splin.norm(Dinit, axis=0)[sp.newaxis, :]

        for i in range(self.max_iter):
            X = self._transform(D, Y)
            e = splin.norm(Y - D.dot(X))
            if e < self.tol:
                break
            D, X = self._update_dict(Y, D, X)

        return D, X

    def fit(
        self, training_feats, labels, Dinit=None,
    ):

        """
        Input
        ----------
        training_feats  -training features (shape = [n_features, n_samples])
        labels          -label matrix for training feature (numberred from 1 to nb of classes)
        Dinit           -initial guess for dictionary
        """

        H_train = sp.zeros((int(labels.max()), training_feats.shape[1]), dtype=float)
        for c in range(int(labels.max())):
            H_train[c, labels == (c + 1)] = 1.0

        W = np.concatenate((training_feats, H_train), axis=0)
        P, X = self._ksvd_fit(W, Dinit)
        self._D = P[: training_feats.shape[0], :]
        self._C = P[training_feats.shape[0] :, :]
        self._D /= splin.norm(self._D, axis=0)[sp.newaxis, :]
        self._C /= splin.norm(self._C, axis=0)[sp.newaxis, :]

    def predict(self, Y):
        X = self._transform(self._D, Y)
        L = sp.dot(self._C, X)
        return L.argmax(L, 0)
