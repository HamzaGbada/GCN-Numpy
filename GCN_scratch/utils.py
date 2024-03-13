import numpy as np


class GraphUtils:
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
    def loss_function(y: np.ndarray, y_hat: np.ndarray) -> np.float_:
        """
        Compute the cross-entropy loss.

            Args:
            Y (numpy.ndarray): True labels, shape (N, C)
            Y_hat (numpy.ndarray): Predicted probabilities, shape (N, C)

            Returns:
            float: Cross-entropy loss
        """
        return -np.sum(np.dot(y, np.log(y_hat))) / y.shape[0]


if __name__ == "__main__":
    adjacency_matrix = np.array(
        [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]]
    )
    print(GraphUtils.degree_matrix(adjacency_matrix))
