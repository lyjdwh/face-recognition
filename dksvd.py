import scipy as sp
import scipy.linalg as splin
from sklearn.linear_model import orthogonal_mp_gram


class DKSVD(object):
    def __init__(
        self, n_components, max_iter=10, tol=1e-6, transform_n_nonzero_coefs=None
    ):
        """
    Parameters
    ----------
    n_components:
        Number of dictionary elements
    max_iter:
        Maximum number of iterations
    tol:
        tolerance for error
    transform_n_nonzero_coefs:
        Number of nonzero coefficients to target
    """
        self.D_ = None
        self.X_ = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs

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

    def fit(self, Y, Dinit=None):
        """
    Use data to learn dictionary and activations.
    Parameters
    ----------
    Y: data. (shape = [n_features, n_samples])
    Dinit: initialization of dictionary. (shape = [n_features, n_components])
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

        self.D_ = D
        self.X_ = X
        return self

    def transform(self, X):
        return self._transform(self.D_, X)
