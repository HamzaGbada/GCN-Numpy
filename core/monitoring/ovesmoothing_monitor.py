import matplotlib.pyplot as plt

from core.metrics.oversmoothing import OversmoothingMetrics


class OversmoothingMonitor:
    """
    Tracks Dirichlet Energy and Rayleigh Quotient, pairwise distance

    across layers or epochs.
    """

    def __init__(self, A):
        self.L = OversmoothingMetrics.normalized_laplacian(A)

        self.de_history = []
        self.rq_history = []
        self.norm_history = []
        self.pairwise_history = []

    def measure(self, X):
        """
        Measure oversmoothing metrics for given features.
        """

        de = OversmoothingMetrics.dirichlet_energy(X, self.L)
        rq = OversmoothingMetrics.rayleigh_quotient(X, self.L)
        norm = OversmoothingMetrics.feature_norm(X)
        pairewise = OversmoothingMetrics.pairwise_distance(X)

        self.de_history.append(de)
        self.rq_history.append(rq)
        self.norm_history.append(norm)
        self.pairwise_history.append(pairewise)

    def reset(self):
        self.de_history = []
        self.rq_history = []
        self.norm_history = []
        self.pairwise_history = []

    def plot(self):

        layers = range(len(self.rq_history))

        plt.figure(figsize=(10,4))

        plt.subplot(1,3,1)
        plt.plot(layers, self.de_history)
        plt.title("Dirichlet Energy")
        plt.xlabel("Layer")
        plt.ylabel("Energy")

        plt.subplot(1,3,2)
        plt.plot(layers, self.rq_history)
        plt.title("Rayleigh Quotient")
        plt.xlabel("Layer")
        plt.ylabel("RQ")

        plt.subplot(1,3,3)
        plt.plot(layers, self.pairwise_history)
        plt.title("Pairwise Distance")
        plt.xlabel("Layer")
        plt.ylabel("PW")
        plt.tight_layout()
        plt.show()