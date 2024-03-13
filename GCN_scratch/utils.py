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

if __name__ == "__main__":
    adjacency_matrix = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0]
    ])
    print(GraphUtils.degree_matrix(adjacency_matrix))
