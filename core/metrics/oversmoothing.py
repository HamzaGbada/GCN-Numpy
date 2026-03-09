"""
Oversmoothing Metrics for Graph Neural Networks.

Numerically stable implementations of Dirichlet Energy, Rayleigh Quotient,
pairwise node-embedding distance, and feature variance — designed for
analysing deep GNNs (32–64 layers) without overflow or NaN propagation.

All computations use float64 precision.
"""

import numpy as np

_EPS = 1e-12  # global epsilon for numerical guards


class OversmoothingMetrics:
    """Static collection of oversmoothing-related graph metrics."""

    # ------------------------------------------------------------------
    # Laplacian
    # ------------------------------------------------------------------
    @staticmethod
    def normalized_laplacian(A: np.ndarray) -> np.ndarray:
        """Compute the symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.

        Guards against zero-degree (isolated) nodes by adding *eps* to the
        degree vector before inversion, which avoids division-by-zero while
        keeping the Laplacian structure essentially unchanged for connected
        components.

        Args:
            A: Adjacency matrix of shape (N, N), dtype will be cast to float64.

        Returns:
            Normalized Laplacian L of shape (N, N), dtype float64.
        """
        A = np.asarray(A, dtype=np.float64)
        deg = np.sum(A, axis=1)  # (N,)
        deg_inv_sqrt = 1.0 / np.sqrt(deg + _EPS)  # guard isolated nodes
        D_inv_sqrt = np.diag(deg_inv_sqrt)
        L = np.eye(A.shape[0], dtype=np.float64) - D_inv_sqrt @ A @ D_inv_sqrt
        return L

    # ------------------------------------------------------------------
    # Dirichlet Energy
    # ------------------------------------------------------------------
    @staticmethod
    def dirichlet_energy(X: np.ndarray, L: np.ndarray) -> np.float64:
        """Dirichlet Energy:  DE = (1/N) * tr(X^T L X).

        Normalised by the number of nodes so that values remain comparable
        across graphs of different sizes.

        Args:
            X: Node feature matrix (N, F).
            L: Normalized Laplacian (N, N).

        Returns:
            Scalar Dirichlet Energy (float64).
        """
        X = np.asarray(X, dtype=np.float64)
        L = np.asarray(L, dtype=np.float64)
        N = X.shape[0]
        return np.float64(np.trace(X.T @ L @ X) / N)

    # ------------------------------------------------------------------
    # Rayleigh Quotient
    # ------------------------------------------------------------------
    @staticmethod
    def rayleigh_quotient(X: np.ndarray, L: np.ndarray) -> np.float64:
        """Rayleigh Quotient:  RQ = tr(X^T L X) / ||X||_F^2.

        Args:
            X: Node feature matrix (N, F).
            L: Normalized Laplacian (N, N).

        Returns:
            Scalar Rayleigh Quotient (float64).
        """
        X = np.asarray(X, dtype=np.float64)
        L = np.asarray(L, dtype=np.float64)
        de = np.trace(X.T @ L @ X)
        norm_sq = np.sum(X ** 2)
        return np.float64(de / (norm_sq + _EPS))

    # ------------------------------------------------------------------
    # Pairwise Distance (vectorised)
    # ------------------------------------------------------------------
    @staticmethod
    def pairwise_distance(X: np.ndarray) -> np.float64:
        """Mean pairwise Euclidean distance between all node embeddings.

        Fully vectorised implementation using the identity
        ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i^T x_j
        followed by a safe square-root.

        Args:
            X: Node feature matrix (N, F).

        Returns:
            Scalar mean pairwise distance (float64).
        """
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        # ||x_i||^2   shape (N, 1)
        sq_norms = np.sum(X ** 2, axis=1, keepdims=True)
        # Squared distance matrix  (N, N)
        D2 = sq_norms + sq_norms.T - 2.0 * (X @ X.T)
        # Numerical guard: clamp negatives caused by floating-point rounding
        np.clip(D2, 0.0, None, out=D2)
        D = np.sqrt(D2)
        return np.float64(np.sum(D) / (N * N))

    # ------------------------------------------------------------------
    # Feature Norm (Frobenius)
    # ------------------------------------------------------------------
    @staticmethod
    def feature_norm(X: np.ndarray) -> np.float64:
        """Squared Frobenius norm of the feature matrix: ||X||_F^2.

        Args:
            X: Node feature matrix (N, F).

        Returns:
            Scalar Frobenius norm squared (float64).
        """
        X = np.asarray(X, dtype=np.float64)
        return np.float64(np.sum(X ** 2))

    # ------------------------------------------------------------------
    # Feature Variance  (NEW)
    # ------------------------------------------------------------------
    @staticmethod
    def feature_variance(X: np.ndarray) -> np.float64:
        """Mean per-feature variance across nodes.

        Measures how diverse the node representations are along each feature
        dimension.  A value collapsing toward zero signals oversmoothing.

        Args:
            X: Node feature matrix (N, F).

        Returns:
            Scalar mean feature variance (float64).
        """
        X = np.asarray(X, dtype=np.float64)
        return np.float64(np.mean(np.var(X, axis=0)))