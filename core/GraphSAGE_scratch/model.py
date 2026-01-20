import numpy as np
from core.utils import GraphUtils
from core.GraphSAGE_scratch.layers import GraphSAGELayer


class GraphSAGE:
    def __init__(self, in_feat, hid_feat, out_feat, layers=1):
        self.init_layer = GraphSAGELayer(in_feat, hid_feat)
        self.hidden_layer = GraphSAGELayer(hid_feat, hid_feat)
        self.out_layer = GraphSAGELayer(hid_feat, out_feat)
        self.layers = layers

    def forward(self, X: np.ndarray, A: np.ndarray):
        H = self.init_layer.forward(X, A)

        for _ in range(self.layers):
            H = self.hidden_layer.forward(H, A)

        logits = self.out_layer.forward(H, A)
        return GraphUtils.softmax(logits)

    def backward(self, y: np.ndarray, y_hat: np.ndarray, lr=0.01):
        """
        y: true labels (N, C)
        y_hat: predicted probabilities (N, C)
        """
        # Softmax + cross-entropy gradient
        grad = (y_hat - y) / y.shape[0]

        grad_out = self.out_layer.backward(grad, lr)
        grad_hid = self.hidden_layer.backward(grad_out, lr)
        grad_init = self.init_layer.backward(grad_hid, lr)

        return grad_init, grad_hid, grad_out
