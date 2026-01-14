import numpy as np

from core.GCN_scratch.layers import GCNLayer
from core.utils import GraphUtils


class GCN:
    def __init__(self, in_feat, hid_feat, out_feat, layers=1):
        self.init_layer = GCNLayer(in_feat, hid_feat)
        self.hidden_layer = GCNLayer(hid_feat, hid_feat)
        self.out_layer = GCNLayer(hid_feat, out_feat)
        self.layers = layers

    def forward(self, X: np.ndarray, A: np.ndarray):
        H = GraphUtils.ReLU(self.init_layer.forward(X, A))
        for i in range(self.layers):
            H = GraphUtils.ReLU(self.hidden_layer.forward(H, A))
        return GraphUtils.softmax(self.out_layer.forward(H, A))

    def backward(self, y, y_hat, alpha=0.01):
        error = (y_hat - y) / y.shape[0]
        gradient_out = self.out_layer.backward(error, alpha)
        gradient_hid = self.hidden_layer.backward(gradient_out, alpha)
        gradient_init = self.init_layer.backward(gradient_hid, alpha)

        return gradient_init, gradient_hid, gradient_out
