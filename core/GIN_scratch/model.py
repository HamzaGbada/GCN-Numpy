import numpy as np

from core.GIN_scratch.layers import GINLayer
from core.utils import GraphUtils


class GIN:
    def __init__(self, in_feat, hid_feat, out_feat, layers=1):
        self.init_layer = GINLayer(in_feat, hid_feat)
        self.hidden_layer = GINLayer(hid_feat, hid_feat)
        self.out_layer = GINLayer(hid_feat, out_feat)

        self.layers = layers

    def forward(self, X: np.ndarray, A: np.ndarray):
        H = GraphUtils.ReLU(self.init_layer.forward(X, A))

        for _ in range(self.layers):
            H = GraphUtils.ReLU(self.hidden_layer.forward(H, A))

        logits = self.out_layer.forward(H, A)
        return GraphUtils.softmax(logits)

    def backward(self, y, y_hat, lr=0.01):
        # softmax + cross entropy gradient
        error = (y_hat - y) / y.shape[0]

        grad_out = self.out_layer.backward(error, lr)
        grad_hid = self.hidden_layer.backward(grad_out, lr)
        grad_init = self.init_layer.backward(grad_hid, lr)

        return grad_init, grad_hid, grad_out