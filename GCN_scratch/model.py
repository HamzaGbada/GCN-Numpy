import numpy as np

from GCN_scratch.layers import GCNLayer
from GCN_scratch.utils import GraphUtils


class GCN:

    def __init__(self, in_feat, hid_feat, out_feat, layers=2):
        self.init_layer = GCNLayer(in_feat, hid_feat)
        self.hidden_layer = GCNLayer(hid_feat, hid_feat)
        self.out_layer = GCNLayer(hid_feat, out_feat)
        self.layers = layers

    def forward(self, X: np.ndarray, A: np.ndarray):
        H = GraphUtils.ReLU(self.init_layer.forward(X, A))
        for i in range(self.layers):
            H = GraphUtils.ReLU(self.hidden_layer.forward(H, A))
        return

