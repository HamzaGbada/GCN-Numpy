"""
GraphSAGE model with residual connections for deep stability.

Supports deep networks via:
  - Independent hidden layers (not weight-shared)
  - Residual connections with mixing coefficient α
  - Row-wise L2 normalization after each hidden layer
  - Gradient clipping in forward to prevent overflow
  - float64 precision throughout
"""

import numpy as np

from core.GraphSAGE_scratch.layers import GraphSAGELayer
from core.utils import GraphUtils


class GraphSAGE:
    """Multi-layer GraphSAGE with independent hidden layers and residual connections.

    Args:
        in_feat: Input feature dimension.
        hid_feat: Hidden feature dimension.
        out_feat: Output (class) dimension.
        layers: Number of *hidden* layers.
        alpha: Residual mixing coefficient  H = α·H_0 + (1-α)·H_new.
    """

    def __init__(self, in_feat: int, hid_feat: int, out_feat: int,
                 layers: int = 1, alpha: float = 0.8):
        self.init_layer = GraphSAGELayer(in_feat, hid_feat)
        self.hidden_layers = [GraphSAGELayer(hid_feat, hid_feat) for _ in range(layers)]
        self.out_layer = GraphSAGELayer(hid_feat, out_feat)
        self.layers = layers
        self.alpha = alpha

    def forward(self, X: np.ndarray, A: np.ndarray,
                return_embeddings: bool = False):
        """Forward pass.

        Args:
            X: Node features (N, F_in).
            A: Adjacency matrix (N, N).
            return_embeddings: If True, return (output, embeddings_list).

        Returns:
            Softmax probabilities (N, C), or (probs, list[np.ndarray]).
        """
        embeddings: list[np.ndarray] = []

        H = self.init_layer.forward(X, A)
        H = np.clip(H, -1e2, 1e2)
        embeddings.append(H)

        H_0 = H
        for layer in self.hidden_layers:
            H_new = layer.forward(H, A)
            H = self.alpha * H_0 + (1.0 - self.alpha) * H_new
            H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-6)
            embeddings.append(H)

        logits = self.out_layer.forward(H, A)
        out = GraphUtils.softmax(logits)

        if return_embeddings:
            return out, embeddings
        return out

    def backward(self, y: np.ndarray, y_hat: np.ndarray, lr: float = 0.01):
        """Backward pass (softmax + cross-entropy gradient)."""
        grad = (y_hat - y) / y.shape[0]

        grad = self.out_layer.backward(grad, lr)
        for layer in reversed(self.hidden_layers):
            grad = layer.backward(grad, lr)
        grad = self.init_layer.backward(grad, lr)
        return grad
