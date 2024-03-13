import numpy as np

from GCN_scratch.utils import GraphUtils


class GCNLayer:
    def __init__(self, in_feat, out_feat):
        self.weight = np.random.randn(in_feat, out_feat)

    def forward(self, X: np.ndarray, A: np.ndarray):
        L = GraphUtils.normalized_graph_laplacien(A)
        return GraphUtils.ReLU(np.dot(np.dot(L, X), self.weight))

