"""
Graph Attention Network (GAT) model with residual connections for deep stability.

Supports deep networks via:
  - Residual connections with mixing coefficient α
  - Row-wise L2 normalization after each hidden layer
  - Gradient clipping in forward to prevent overflow
  - float64 precision throughout
"""

import numpy as np

from core.GAT_scratch.layers import GATLayer
from core.utils import GraphUtils


class GAT:
    """Multi-layer GAT with residual connections for deep-network stability.

    Args:
        in_feat: Input feature dimension.
        hid_feat: Hidden feature dimension.
        out_feat: Output (class) dimension.
        layers: Number of *hidden* layers.
        alpha: Residual mixing coefficient  H = α·H_0 + (1-α)·H_new.
    """

    def __init__(self, in_feat: int, hid_feat: int, out_feat: int,
                 layers: int = 1, alpha: float = 0.8):
        self.init_layer = GATLayer(in_feat, hid_feat)
        self.hidden_layers = [GATLayer(hid_feat, hid_feat) for _ in range(layers)]
        self.out_layer = GATLayer(hid_feat, out_feat)
        self.layers = layers
        self.alpha = alpha

        # ReLU masks for backward
        self.relu_masks: list[np.ndarray] = []

    def forward(self, X: np.ndarray, A: np.ndarray,
                return_embeddings: bool = False):
        """Forward pass.

        Args:
            X: Node features (N, F_in).
            A: Adjacency matrix (N, N).
            return_embeddings: If True, return (logits, embeddings_list).

        Returns:
            Logits (N, C), or (logits, list[np.ndarray]).
        """
        self.relu_masks = []
        embeddings: list[np.ndarray] = []

        # --- init layer ---
        H = self.init_layer.forward(X, A)
        mask = H > 0
        self.relu_masks.append(mask)
        H = H * mask
        H = np.clip(H, -1e2, 1e2)
        embeddings.append(H)

        # --- hidden layers with residual ---
        H_0 = H
        for layer in self.hidden_layers:
            H_new = layer.forward(H, A)
            mask = H_new > 0
            self.relu_masks.append(mask)
            H_new = H_new * mask
            # Residual + normalisation
            H = self.alpha * H_0 + (1.0 - self.alpha) * H_new
            H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-6)
            embeddings.append(H)

        # --- output layer ---
        logits = self.out_layer.forward(H, A)

        if return_embeddings:
            return logits, embeddings
        return logits

    def backward(self, y: np.ndarray, y_hat: np.ndarray, lr: float = 0.01):
        """Backward pass (softmax + cross-entropy gradient)."""
        N = y.shape[0]
        grad = (y_hat - y) / N

        grad = self.out_layer.backward(grad, lr)

        for layer, relu_mask in zip(
            reversed(self.hidden_layers), reversed(self.relu_masks[1:])
        ):
            grad = grad * relu_mask
            grad = layer.backward(grad, lr)

        grad = grad * self.relu_masks[0]
        grad = self.init_layer.backward(grad, lr)
        return grad
