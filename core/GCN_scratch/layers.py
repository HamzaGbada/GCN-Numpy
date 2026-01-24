import numpy as np

from core.MPNN_scratch.base import MPNNLayer
from core.utils import GraphUtils


class GCNLayer(MPNNLayer):
    """
    Graph Convolutional Network layer.
    
    Implements GCN as a specialization of MPNN:
    - message(): Identity (pass through features)
    - aggregate(): Normalized Laplacian multiplication L @ X
    - update(): Linear transformation W @ aggregated + bias
    """
    
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.weight = np.random.randn(in_feat, out_feat)
        self.bias = np.zeros((1, out_feat))
        # Cache for backward
        self.L = None

    def message(self, X, A):
        """Identity message - features passed through unchanged."""
        return X
    
    def aggregate(self, messages, A):
        """Apply normalized graph Laplacian transformation."""
        self.L = GraphUtils.normalized_graph_laplacien(A)
        return np.dot(self.L, messages)
    
    def update(self, aggregated):
        """Apply linear transformation."""
        return np.dot(aggregated, self.weight) + self.bias

    def backward(self, error: np.ndarray, lr):
        """
        Backward pass for GCN layer.
        
        Args:
            error: Gradient of loss with respect to output
            lr: Learning rate
            
        Returns:
            Gradient with respect to input features
        """
        feat = np.dot(self.X.T, self.L)
        grad_weight = np.dot(feat, error)
        grad_bias = np.sum(error, axis=0, keepdims=True)
        grad_input = np.dot(self.L.T, np.dot(error, self.weight.T))

        self.weight -= lr * grad_weight
        self.bias -= lr * grad_bias
        return grad_input
