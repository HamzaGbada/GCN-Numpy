import numpy as np

from core.MPNN_scratch.base import MPNNLayer
from core.utils import GraphUtils, BatchNorm1D


class GINLayer(MPNNLayer):
    """
    Graph Isomorphism Network layer.
    
    Implements GIN as a specialization of MPNN:
    - message(): Identity (pass through features)
    - aggregate(): Sum aggregation A @ H
    - update(): Apply (1+eps)*H + M and MLP with batch normalization
    """
    
    def __init__(self, in_feat, out_feat, eps=0.0, learn_eps=True):
        super().__init__()
        # MLP parameters (1 hidden layer MLP)
        self.W1 = np.random.randn(in_feat, out_feat)
        self.b1 = np.zeros((1, out_feat))

        self.W2 = np.random.randn(out_feat, out_feat)
        self.b2 = np.zeros((1, out_feat))

        self.bn = BatchNorm1D(out_feat)
        # epsilon
        self.eps = eps
        self.learn_eps = learn_eps

        # Additional caches for backward
        self.H = None
        self.M = None
        self.S = None
        self.Z1 = None
        self.A1 = None

    def message(self, H, A):
        """Identity message - features passed through unchanged."""
        self.H = H
        return H
    
    def aggregate(self, messages, A):
        """Sum aggregation: A @ H."""
        self.M = np.dot(A, messages)
        return self.M
    
    def update(self, aggregated):
        """
        Apply GIN update: (1+eps)*H + M, then MLP with batch norm.
        """
        # Sum aggregation with self-loop handled via epsilon
        self.S = (1 + self.eps) * self.H + aggregated

        # MLP with batch Normalization
        self.Z1 = np.dot(self.S, self.W1) + self.b1
        Z1_bn = self.bn.forward(self.Z1)
        self.A1 = GraphUtils.ReLU(Z1_bn)
        out = np.dot(self.A1, self.W2) + self.b2

        return out

    def backward(self, grad_out: np.ndarray, lr):
        """
        Backward pass for GIN layer.
        
        Args:
            grad_out: dL / dH^{l+1}
            lr: Learning rate
            
        Returns:
            Gradient with respect to input features
        """
        # ---- MLP backward ----
        grad_W2 = np.dot(self.A1.T, grad_out)
        grad_b2 = np.sum(grad_out, axis=0, keepdims=True)

        grad_A1 = np.dot(grad_out, self.W2.T)
        grad_Z1_bn = grad_A1 * (self.A1 > 0)

        grad_Z1 = self.bn.backward(grad_Z1_bn, lr)

        grad_W1 = np.dot(self.S.T, grad_Z1)
        grad_b1 = np.sum(grad_Z1, axis=0, keepdims=True)

        grad_S = np.dot(grad_Z1, self.W1.T)

        # ---- epsilon gradient ----
        if self.learn_eps:
            grad_eps = np.sum(grad_S * self.H)
            self.eps -= lr * grad_eps

        # ---- gradient w.r.t input H ----
        grad_H = (1 + self.eps) * grad_S + np.dot(self.A.T, grad_S)

        # ---- update parameters ----
        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2

        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1

        return grad_H
