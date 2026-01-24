import numpy as np

from core.MPNN_scratch.base import MPNNLayer
from core.utils import GraphUtils


class GraphSAGELayer(MPNNLayer):
    """
    GraphSAGE layer.
    
    Implements GraphSAGE as a specialization of MPNN:
    - message(): Identity (pass through features)
    - aggregate(): Mean aggregation D^{-1} @ A @ X
    - update(): Apply X @ W_self + H_neigh @ W_neigh + bias with ReLU
    """
    
    def __init__(self, in_feat, out_feat):
        super().__init__()
        # Separate weights (canonical GraphSAGE)
        self.W_self = np.random.randn(in_feat, out_feat) * 0.01
        self.W_neigh = np.random.randn(in_feat, out_feat) * 0.01
        self.bias = np.zeros((1, out_feat))

        # Additional caches for backward
        self.H_neigh = None
        self.Z = None

    def message(self, X, A):
        """Identity message - features passed through unchanged."""
        return X
    
    def aggregate(self, messages, A):
        """Mean aggregation: D^{-1} A X."""
        D = np.sum(A, axis=1, keepdims=True) + 1e-8
        A_norm = A / D
        self.H_neigh = np.dot(A_norm, messages)
        return self.H_neigh
    
    def update(self, aggregated):
        """Apply linear transformation with separate self and neighbor weights, then ReLU."""
        # Linear transformation
        self.Z = np.dot(self.X, self.W_self) + np.dot(aggregated, self.W_neigh) + self.bias
        return GraphUtils.ReLU(self.Z)

    def backward(self, grad_out: np.ndarray, lr: float):
        """
        Backward pass for GraphSAGE layer.
        
        Args:
            grad_out: dL/dH^{(l+1)}
            lr: Learning rate
            
        Returns:
            Gradient with respect to input features
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
