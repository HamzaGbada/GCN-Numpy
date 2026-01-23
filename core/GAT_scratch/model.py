import numpy as np
from core.GAT_scratch.layers import GATLayer

class GAT:
    def __init__(self, in_feat, hid_feat, out_feat, layers=1):
        # Input layer
        self.init_layer = GATLayer(in_feat, hid_feat)

        # Hidden layers (list of independent GAT layers)
        self.hidden_layers = [
            GATLayer(hid_feat, hid_feat)
            for _ in range(layers)
        ]

        # Output layer
        self.out_layer = GATLayer(hid_feat, out_feat)

        # Store ReLU masks for backward
        self.relu_masks = []

    def forward(self, X: np.ndarray, A: np.ndarray):
        """
        X: Node features (N, F_in)
        A: Adjacency matrix (N, N)
        """
        self.relu_masks = []

        # --- Input layer ---
        H = self.init_layer.forward(X, A)
        mask = (H > 0)
        self.relu_masks.append(mask)
        H = H * mask  # ReLU

        # --- Hidden layers ---
        for layer in self.hidden_layers:
            H = layer.forward(H, A)
            mask = (H > 0)
            self.relu_masks.append(mask)
            H = H * mask  # ReLU

        # --- Output layer ---
        logits = self.out_layer.forward(H, A)
        return logits  # No softmax here; handled in loss

    def backward(self, y: np.ndarray, y_hat: np.ndarray, lr=0.01):
        """
        Backward pass using softmax + cross-entropy gradient:
        dL/d(logits) = (y_hat - y)/N

        y: One-hot labels (N, C)
        y_hat: softmax outputs from forward (N, C)
        lr: learning rate
        """
        N = y.shape[0]

        # 1. Softmax + cross-entropy gradient
        grad = (y_hat - y) / N  # (N, C)

        # 2. Output layer backward
        grad = self.out_layer.backward(grad, lr)

        # 3. Hidden layers backward (reverse order)
        for layer, relu_mask in zip(
            reversed(self.hidden_layers),
            reversed(self.relu_masks[1:])
        ):
            grad = grad * relu_mask  # ReLU backward
            grad = layer.backward(grad, lr)

        # 4. Input layer backward
        grad = grad * self.relu_masks[0]  # ReLU backward
        grad = self.init_layer.backward(grad, lr)

        return grad
