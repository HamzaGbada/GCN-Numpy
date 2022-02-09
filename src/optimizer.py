import numpy as np


class Gradient_Descent_Optimizer():
    def __init__(self, alpha, w):
        self.alpha = alpha
        self.w = w
        self._y_pred = None
        self._y_true = None
        self._out = None
        self.bs = None
        self.train_nodes = None

    def __call__(self, y_pred, y_true, train_nodes=None):
        self.y_pred = y_pred
        self.y_true = y_true

        if self.train_nodes is None:
            self.train_nodes = np.arange(y_pred.shape[0])
        else:
            self.train_nodes = train_nodes

        self.bs = self.train_nodes.shape[0]

    @property
    def out(self):
        return self._out

    @out.setter
    def out(self, y):
        self._out = y
