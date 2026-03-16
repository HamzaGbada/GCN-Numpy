"""
Graph Convolutional Network (GCN) model with GCNII-style residual connections.

Supports deep networks (32–64 layers) via:
  - Residual connections (GCNII initial-residual mixing)
  - Row-wise L2 normalization after each hidden layer
  - Clipping to prevent unbounded growth
  - float64 precision throughout
"""

import numpy as np

from core.GCN_scratch.layers import GCNLayer
from core.utils import GraphUtils


class GCN:
    """Multi-layer GCN with GCNII-style residual for deep-network stability.

    Args:
        in_feat: Input feature dimension.
        hid_feat: Hidden feature dimension.
        out_feat: Output (class) dimension.
        layers: Number of *hidden* layers (depth).
        alpha: Residual mixing coefficient  H = α·H_0 + (1-α)·H_new.
    """

    def __init__(self, in_feat: int, hid_feat: int, out_feat: int,
                 layers: int = 1, alpha: float = 0.8):
        self.init_layer = GCNLayer(in_feat, hid_feat)
        self.hidden_layers = [GCNLayer(hid_feat, hid_feat) for _ in range(layers)]
        self.out_layer = GCNLayer(hid_feat, out_feat)
        self.layers = layers
        self.alpha = alpha

    def forward(self, X: np.ndarray, A: np.ndarray,
                return_embeddings: bool = False):
        """Forward pass.

        Args:
            X: Node features (N, F_in), dtype float64.
            A: (Normalised) adjacency matrix (N, N).
            return_embeddings: If True, return (output, embeddings_list).

        Returns:
            Softmax class probabilities (N, C), or tuple
            (probabilities, list[np.ndarray]) when *return_embeddings* is True.
        """
        embeddings: list[np.ndarray] = []

        # --- initial projection ---
        H = GraphUtils.ReLU(self.init_layer.forward(X, A))
        H = np.clip(H, -1e2, 1e2)
        embeddings.append(H)

        # --- hidden layers with GCNII residual ---
        H_0 = H  # initial residual anchor
        for layer in self.hidden_layers:
            H_new = GraphUtils.ReLU(layer.forward(H, A))
            H = self.alpha * H_0 + (1.0 - self.alpha) * H_new
            H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-6)
            embeddings.append(H)

        # --- output layer ---
        out = GraphUtils.softmax(self.out_layer.forward(H, A))

        if return_embeddings:
            return out, embeddings
        return out

    def backward(self, y: np.ndarray, y_hat: np.ndarray, lr: float = 0.001):
        """Backward pass (vanilla gradient descent) with gradient clipping."""
        error = (y_hat - y) / y.shape[0]
        grad = self.out_layer.backward(error, lr)
        for layer in reversed(self.hidden_layers):
            grad = layer.backward(grad, lr)
            grad = np.clip(grad, -1.0, 1.0)
        grad = self.init_layer.backward(grad, lr)
        return grad
