import numpy as np


class GraphUtils:
    @staticmethod
    def softmax_cross_entropy(y_hat: np.ndarray, y: np.ndarray):
        """
        y_hat: softmax output (N, C)
        y: one-hot labels (N, C)
        """

        N = y.shape[0]

        # Loss
        loss = -np.sum(y * np.log(y_hat + 1e-9)) / N

        # Gradient w.r.t logits (softmax + CE fused)
        error = (y_hat - y) / N

        return loss, error

    @staticmethod
    def leaky_relu_backward(x: np.ndarray, alpha=0.2):
        """
        Derivative of LeakyReLU with respect to input x.
        Works for scalars and NumPy arrays.
        """
        return np.where(x > 0, 1.0, alpha)

    @staticmethod
    def masked_softmax(e: np.ndarray, A: np.ndarray) -> np.ndarray:
        e = e - np.max(e, axis=1, keepdims=True)
        exp_e = np.exp(e) * A
        return exp_e / (np.sum(exp_e, axis=1, keepdims=True) + 1e-9)

    @staticmethod
    def leaky_relu(x: np.ndarray, alpha=0.2) -> np.ndarray:
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def degree_matrix(A: np.ndarray) -> np.ndarray:
        """
        A: np.ndarray is the adjacency matrix of the graph.
        """
        return np.diag(np.sum(A, axis=1))

    @staticmethod
    def normalized_graph_laplacien(A: np.ndarray) -> np.ndarray:
        """
        A (np.ndarray) is the adjacency matrix of the graph.
        """
        degree = GraphUtils.degree_matrix(A)
        d = np.linalg.inv(np.sqrt(degree))
        return np.identity(A.shape[0]) - np.dot(d, np.dot(A, d))

    @staticmethod
    def ReLU(x: np.ndarray) -> np.ndarray:
        return x * (x > 0)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax values for each sets of scores in x."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    @staticmethod
    def loss_function(y: np.ndarray, y_hat: np.ndarray) -> np.float64:
        """
        Compute the cross-entropy loss.

            Args:
            Y (numpy.ndarray): True labels, shape (N, C)
            Y_hat (numpy.ndarray): Predicted probabilities, shape (N, C)

            Returns:
            float: Cross-entropy loss
        """
        return -np.mean(np.sum(y * np.log(y_hat + 1e-8), axis=1))
        # return -np.sum(y * np.log(y_hat)) / y.shape[0]


class BatchNorm1D:
    def __init__(self, dim, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum

        # learnable parameters
        self.gamma = np.ones((1, dim))
        self.beta = np.zeros((1, dim))

        # running stats (for inference, optional)
        self.running_mean = np.zeros((1, dim))
        self.running_var = np.ones((1, dim))

        # cache
        self.X = None
        self.mean = None
        self.var = None

    def forward(self, X, training=True):
        """
        X: (N, F)
        """
        if training:
            self.mean = X.mean(axis=0, keepdims=True)
            self.var = X.var(axis=0, keepdims=True)

            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * self.var
            )
        else:
            self.mean = self.running_mean
            self.var = self.running_var

        self.X = X
        X_hat = (X - self.mean) / np.sqrt(self.var + self.eps)
        out = self.gamma * X_hat + self.beta
        return out

    def backward(self, grad_out, lr):
        """
        grad_out: (N, F)
        """
        N = grad_out.shape[0]

        X_hat = (self.X - self.mean) / np.sqrt(self.var + self.eps)

        dgamma = np.sum(grad_out * X_hat, axis=0, keepdims=True)
        dbeta = np.sum(grad_out, axis=0, keepdims=True)

        dX_hat = grad_out * self.gamma
        dvar = np.sum(
            dX_hat * (self.X - self.mean) * -0.5 * (self.var + self.eps) ** (-1.5),
            axis=0,
            keepdims=True,
        )
        dmean = np.sum(
            dX_hat * -1 / np.sqrt(self.var + self.eps), axis=0, keepdims=True
        ) + dvar * np.mean(-2 * (self.X - self.mean), axis=0, keepdims=True)

        dX = (
            dX_hat / np.sqrt(self.var + self.eps)
            + dvar * 2 * (self.X - self.mean) / N
            + dmean / N
        )

        # update
        self.gamma -= lr * dgamma
        self.beta -= lr * dbeta

        return dX
