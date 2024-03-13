import numpy as np

from GCN_scratch.utils import GraphUtils


class GCNLayer:
    def __init__(self, in_feat, out_feat):
        self.weight = np.random.randn(in_feat, out_feat)
        self.bias = np.zeros((1, out_feat))
        self.A = None
        self.X = None

    def forward(self, X: np.ndarray, A: np.ndarray):
        self.A = A
        self.X = X
        L = GraphUtils.normalized_graph_laplacien(A)
        return np.dot(np.dot(L, X), self.weight) + self.bias

    def backward(self, error: np.ndarray, lr):
        L = GraphUtils.normalized_graph_laplacien(self.A)
        feat = np.dot(self.X.T, L)
        grad_weight = np.dot(feat, error)
        grad_bias = np.sum(error, axis=0, keepdims=True)
        print(f"shape of L.T {L.T.shape}")
        print(f"shape of self.weight.T {self.weight.T.shape}")
        print(f"shape of error {error.shape}")
        grad_input = np.dot(L.T, np.dot(error, self.weight.T))

        self.weight -= lr * grad_weight
        self.bias -= lr * grad_bias
        return grad_input
