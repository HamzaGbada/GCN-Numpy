import numpy as np

from core.GCN_scratch.layers import GCNLayer
from core.utils import GraphUtils


import numpy as np
from core.GCN_scratch.layers import GCNLayer
from core.utils import GraphUtils

class GCN:
    def __init__(self, in_feat, hid_feat, out_feat, layers=1, alpha=0.8):
        self.init_layer = GCNLayer(in_feat, hid_feat)
        self.hidden_layers = [GCNLayer(hid_feat, hid_feat) for _ in range(layers)]
        self.out_layer = GCNLayer(hid_feat, out_feat)
        self.layers = layers
        self.alpha = alpha  # GCNII residual

    def forward(self, X, A, return_embeddings=False):
        embeddings = []

        H = GraphUtils.ReLU(self.init_layer.forward(X, A))
        H = np.clip(H, -1e2, 1e2)
        embeddings.append(H)

        for layer in self.hidden_layers:
            H_new = GraphUtils.ReLU(layer.forward(H, A))
            # GCNII-style update
            H = self.alpha * H + (1 - self.alpha) * H_new
            H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-6)
            embeddings.append(H)

        out = GraphUtils.softmax(self.out_layer.forward(H, A))
        if return_embeddings:
            return out, embeddings
        return out

    def backward(self, y, y_hat, lr=0.001):
        error = (y_hat - y) / y.shape[0]
        grad = self.out_layer.backward(error, lr)
        for layer in reversed(self.hidden_layers):
            grad = layer.backward(grad, lr)
        grad = self.init_layer.backward(grad, lr)
        return grad
