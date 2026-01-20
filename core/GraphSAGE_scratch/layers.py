import numpy as np
from core.utils import GraphUtils


class GraphSAGELayer:
    def __init__(self, in_feat, out_feat):
        # Separate weights (canonical GraphSAGE)
        self.W_self = np.random.randn(in_feat, out_feat) * 0.01
        self.W_neigh = np.random.randn(in_feat, out_feat) * 0.01
        self.bias = np.zeros((1, out_feat))

        # Cache for backward
        self.X = None
        self.A = None
        self.H_neigh = None
        self.Z = None

    def forward(self, X: np.ndarray, A: np.ndarray):
        """
        X: (N, Fin)
        A: (N, N) adjacency matrix
        """
        self.X = X
        self.A = A

        # Mean aggregation: D^{-1} A X
        D = np.sum(A, axis=1, keepdims=True) + 1e-8
        A_norm = A / D
        self.H_neigh = np.dot(A_norm, X)

        # Linear transformation
        self.Z = (
            np.dot(X, self.W_self)
            + np.dot(self.H_neigh, self.W_neigh)
            + self.bias
        )

        return GraphUtils.ReLU(self.Z)

    def backward(self, grad_out: np.ndarray, lr: float):
        """
        grad_out: dL/dH^{(l+1)}
        """
        # ReLU gradient
        grad_Z = grad_out * (self.Z > 0)

        # Gradients w.r.t parameters
        grad_W_self = np.dot(self.X.T, grad_Z)
        grad_W_neigh = np.dot(self.H_neigh.T, grad_Z)
        grad_bias = np.sum(grad_Z, axis=0, keepdims=True)

        # Gradient w.r.t input features
        grad_X_self = np.dot(grad_Z, self.W_self.T)
        grad_X_neigh = np.dot(grad_Z, self.W_neigh.T)

        # Backprop through mean aggregation
        D = np.sum(self.A, axis=1, keepdims=True) + 1e-8
        A_norm = self.A / D
        grad_X_from_neigh = np.dot(A_norm.T, grad_X_neigh)

        grad_X = grad_X_self + grad_X_from_neigh

        # Parameter update
        self.W_self -= lr * grad_W_self
        self.W_neigh -= lr * grad_W_neigh
        self.bias -= lr * grad_bias

        return grad_X
