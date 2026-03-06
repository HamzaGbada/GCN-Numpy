import numpy as np


class OversmoothingMetrics:
    """
    Dirichlet Energy and Rayleigh Quotient metrics
    used to measure GNN oversmoothing.
    """

    @staticmethod
    def normalized_laplacian(A):
        """
        Compute normalized Laplacian:
        L = I - D^{-1/2} A D^{-1/2}
        """

        deg = np.sum(A, axis=1)
        deg_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-12))

        L = np.eye(A.shape[0]) - deg_inv_sqrt @ A @ deg_inv_sqrt

        return L

    @staticmethod
    def dirichlet_energy(X, L):
        """
        DE = tr(X^T L X)
        """
        return np.trace(X.T @ L @ X)

    @staticmethod
    def feature_norm(X):
        """
        Frobenius norm squared
        """
        return np.sum(X ** 2)

    @staticmethod
    def rayleigh_quotient(X, L):
        """
        RQ = tr(X^T L X) / ||X||_F^2
        """

        de = OversmoothingMetrics.dirichlet_energy(X, L)
        norm = OversmoothingMetrics.feature_norm(X)

        return de / (norm + 1e-12)