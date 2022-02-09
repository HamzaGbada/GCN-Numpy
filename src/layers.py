import numpy as np


class GCN_Layer():
    """
    Graph Convolution Layer
    """

    def __init__(self, nbr_int, nbr_out, activation=None):
        self.nbr_int = nbr_int
        self.nbr_out = nbr_out
        # bias term
        self.W = bias(self.nbr_out, self.nbr_int)
        self.activation = activation

    def forward(self, A, X, W=None):
        """
            An implementation of forward propagation
            H=activation(W.A.H)
            where:
            H is the coming feat vect (intial layer it equal to X)
            W is learnable Weights
            Return a hidden layer output (H)
        """
        # initialization
        # GCN Layer Forward
        self._A = A
        self._X = np.dot(A, X).T
        if W is None:
            W = self.W

        H = np.dot(W, self._X)
        if self.activation is not None:
            H = self.activation(H)
        self._H = H
        return self._H.T

    def backward(self, optim, update=True):
        d_th = 1 - np.asarray(self._H.T) ** 2
        d2 = np.multiply(optim.out, d_th)

        self.grad = np.dot(self._A, np.dot(d2, self.W))
        optim.out = self.grad

        dW = np.asarray(np.dot(d2.T, self._X.T)) / optim.bs
        dW_wd = self.W * optim.w / optim.bs

        if update:
            self.W -= (dW + dW_wd) * optim.alpha

        return dW + dW_wd


# a dense layer
class Softmax_Layer():
    """
    Softmax Layer
    """

    def __init__(self, nbr_in, nbr_out):
        self.nbr_in = nbr_in
        self.nbr_out = nbr_out
        self.W = bias(self.nbr_out, self.nbr_in)
        self.b = np.zeros((self.nbr_out, 1))
        self._X = None  # Used to calculate gradients

    def forward(self, X, W=None, b=None):

        # Softmax forward
        self._X = X.T
        print("Softmax input Shape")
        print(self._X.shape)
        if W is None:
            W = self.W
            print("Softmax W shape")
            print(W.shape)
        if b is None:
            b = self.b
            print("Softmax bias shape")
            print(b.shape)

        proj = np.asarray(np.dot(W, self._X)) + b
        return softmax(proj).T

    def backward(self, optim, update=True):
        train_mask = np.zeros(optim.y_pred.shape[0])
        train_mask = [optim.train_nodes] = 1
        train_mask = train_mask.reshape((-1, 1))

        # le derivé du Cost funtion
        d1 = np.asarray((optim.y_pred - optim.y_true))
        d1 = np.multiply(d1, train_mask)

        self.grad = np.dot(d1, self.W)
        optim.out = self.grad

        dW = np.dot(d1.T, self._X.T) / optim.bs
        db = d1.T.sum(axis=1, deepdims=True) / optim.bs

        dW_ws = self.W * optim.wd / optim.bs

        if update:
            self.w -= (dW + dW_wd) * optim.lr
            self.b -= (db.reshape(self.b.shape) * optim.lr)

        return dW + dW_wd, db.reshape(self.b.shape)
