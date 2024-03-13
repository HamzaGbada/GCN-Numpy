import numpy as np

from GCN_scratch.layers import GCNLayer
from GCN_scratch.utils import GraphUtils


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

    def backward(self, y, y_hat, X: np.ndarray, A: np.ndarray, alpha=0.01):
        loss = GraphUtils.loss_function(y, y_hat)
        feat = np.dot(X.T, A.T)
        error = y_hat - y
        gradient_w = np.dot(feat, error)/A.shape[0]
        gradient_b = np.dot(feat, error)/A.shape[0]
        self.init_layer.weight = self.init_layer.weight - alpha * gradient
        self.hidden_layer.weight = self.init_layer.weight - alpha * gradient
        self.out_layer.weight = self.init_layer.weight - alpha * gradient

    def backward(self, X, A, H1, Y, probs):
        # Compute gradient of loss w.r.t. logits
        dL_dlogits = probs - Y

        # Gradient of loss w.r.t. W2 and b2
        dW2 = np.dot(H1.T, dL_dlogits)
        db2 = np.sum(dL_dlogits, axis=0, keepdims=True)

        # Update weights of output layer
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        # Gradient of loss w.r.t. H1
        dL_dH1 = np.dot(dL_dlogits, self.W2.T)

        # Gradient of loss w.r.t. H1 activation
        dH1_dZ1 = np.where(H1 > 0, 1, 0)

        # Gradient of loss w.r.t. W1 and b1
        dZ1_dW1 = np.dot(np.dot(X.T, A.T), dH1_dZ1)
        dW1 = np.dot(X.T, np.dot(A.T, dL_dH1 * dH1_dZ1))
        db1 = np.sum(dL_dH1 * dH1_dZ1, axis=0, keepdims=True)

        # Update weights of hidden layer
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
