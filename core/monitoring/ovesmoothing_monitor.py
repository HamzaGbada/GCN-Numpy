"""
Oversmoothing Monitor — Metric Collection, Persistence, and Visualisation.

Collects per-layer oversmoothing metrics during training, persists them as
``.npy`` arrays, and generates Matplotlib line-plots and heatmaps for
post-experiment analysis.

All data is stored under a configurable *result_dir* (default
``results/gcn_oversmoothing/``).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt

from core.metrics.oversmoothing import OversmoothingMetrics


class OversmoothingMonitor:
    """Collect, store, and visualise oversmoothing metrics across training.

    Metrics tracked per layer at each recording epoch:
        - Dirichlet Energy
        - Rayleigh Quotient
        - Pairwise Distance
        - Feature Variance

    Additionally, training loss is recorded every epoch.

    Args:
        A: Raw adjacency matrix (N, N).  Used to compute the normalised
           Laplacian once at construction time.
        result_dir: Directory for output files (.npy + .png).
        metric_interval: Record oversmoothing metrics every *metric_interval*
                         epochs (default 5).
    """

    # Canonical ordering of metrics — used for iteration & file naming.
    METRIC_NAMES = [
        "dirichlet_energy",
        "rayleigh",
        "pairwise_distance",
        "feature_variance",
    ]

    def __init__(
        self,
        A: np.ndarray,
        result_dir: str = "results/gcn_oversmoothing",
        metric_interval: int = 5,
    ):
        self.L = OversmoothingMetrics.normalized_laplacian(A)
        self.result_dir = result_dir
        self.metric_interval = metric_interval
        os.makedirs(result_dir, exist_ok=True)

        # Loss is recorded every epoch → flat list.
        self.loss_history: list[float] = []

        # Metric histories are 2-D: metric_history[epoch_idx][layer_idx].
        # Only epochs where metrics are recorded appear.
        self.metric_epochs: list[int] = []  # which epochs were recorded
        self.dirichlet_history: list[list[float]] = []
        self.rayleigh_history: list[list[float]] = []
        self.pairwise_history: list[list[float]] = []
        self.variance_history: list[list[float]] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record_loss(self, loss: float) -> None:
        """Append a single loss value (call every epoch)."""
        self.loss_history.append(float(loss))

    def record_metrics(self, epoch: int, embeddings: list[np.ndarray]) -> None:
        """Compute all four metrics for every layer embedding.

        Should be called every *metric_interval* epochs.

        Args:
            epoch: Current epoch number (0-indexed).
            embeddings: List of node feature matrices, one per layer.
        """
        de_per_layer, rq_per_layer, pw_per_layer, fv_per_layer = [], [], [], []

        for H in embeddings:
            H = np.asarray(H, dtype=np.float64)
            de_per_layer.append(OversmoothingMetrics.dirichlet_energy(H, self.L))
            rq_per_layer.append(OversmoothingMetrics.rayleigh_quotient(H, self.L))
            pw_per_layer.append(OversmoothingMetrics.pairwise_distance(H))
            fv_per_layer.append(OversmoothingMetrics.feature_variance(H))

        self.metric_epochs.append(epoch)
        self.dirichlet_history.append(de_per_layer)
        self.rayleigh_history.append(rq_per_layer)
        self.pairwise_history.append(pw_per_layer)
        self.variance_history.append(fv_per_layer)

    def should_record(self, epoch: int) -> bool:
        """Return True if metrics should be recorded at this epoch."""
        return epoch % self.metric_interval == 0

    # ------------------------------------------------------------------
    # Persistence (.npy)
    # ------------------------------------------------------------------
    def save_npy(self) -> None:
        """Save all collected data as .npy files."""
        np.save(
            os.path.join(self.result_dir, "loss.npy"),
            np.array(self.loss_history, dtype=np.float64),
        )
        for name, history in self._iter_metric_histories():
            np.save(
                os.path.join(self.result_dir, f"{name}.npy"),
                np.array(history, dtype=np.float64),
            )
        print(f"[Monitor] .npy files saved to {self.result_dir}/")

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def _iter_metric_histories(self):
        """Yield (name, history_2d) for each metric."""
        yield "dirichlet_energy", self.dirichlet_history
        yield "rayleigh", self.rayleigh_history
        yield "pairwise_distance", self.pairwise_history
        yield "feature_variance", self.variance_history

    @staticmethod
    def _pretty(name: str) -> str:
        return name.replace("_", " ").title()

    # ------------------------------------------------------------------
    # Line plots
    # ------------------------------------------------------------------
    def plot_loss_curve(self) -> None:
        """Save training-loss curve."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(len(self.loss_history)), self.loss_history, linewidth=1.5)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training Loss", fontsize=14)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(self.result_dir, "loss_curve.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"[Monitor] Saved {path}")

    def plot_metric_vs_layers(self) -> None:
        """Save one figure per metric: metric value vs. layer index.

        Uses the *last* recorded epoch's data (final model state).
        """
        if not self.metric_epochs:
            return
        for name, history in self._iter_metric_histories():
            values = history[-1]  # last recorded epoch
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(range(len(values)), values, marker="o", linewidth=1.5, markersize=4)
            ax.set_xlabel("Layer", fontsize=12)
            ax.set_ylabel(self._pretty(name), fontsize=12)
            ax.set_title(
                f"{self._pretty(name)} vs. Layer  (Epoch {self.metric_epochs[-1]})",
                fontsize=14,
            )
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = os.path.join(self.result_dir, f"{name}_layers.png")
            fig.savefig(path, dpi=200)
            plt.close(fig)
            print(f"[Monitor] Saved {path}")

    def plot_metric_vs_epochs(self) -> None:
        """Save one figure per metric: last-layer metric value vs. epoch."""
        if not self.metric_epochs:
            return
        for name, history in self._iter_metric_histories():
            last_layer_vals = [epoch_data[-1] for epoch_data in history]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(
                self.metric_epochs, last_layer_vals,
                marker="o", linewidth=1.5, markersize=4,
            )
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel(self._pretty(name), fontsize=12)
            ax.set_title(
                f"{self._pretty(name)} (last layer) vs. Epoch", fontsize=14,
            )
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = os.path.join(self.result_dir, f"{name}_epochs.png")
            fig.savefig(path, dpi=200)
            plt.close(fig)
            print(f"[Monitor] Saved {path}")

    # ------------------------------------------------------------------
    # Heatmaps
    # ------------------------------------------------------------------
    def plot_heatmaps(self) -> None:
        """Save Layer × Epoch heatmaps for each metric."""
        if not self.metric_epochs:
            return
        for name, history in self._iter_metric_histories():
            data = np.array(history, dtype=np.float64)  # (n_recorded, n_layers)
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(data.T, aspect="auto", cmap="viridis", origin="lower")
            ax.set_xlabel("Recorded Epoch Index", fontsize=12)
            ax.set_ylabel("Layer", fontsize=12)
            ax.set_title(f"{self._pretty(name)} — Layer × Epoch Heatmap", fontsize=14)

            # Epoch tick labels
            n_ticks = min(len(self.metric_epochs), 10)
            tick_idx = np.linspace(0, len(self.metric_epochs) - 1, n_ticks, dtype=int)
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([str(self.metric_epochs[i]) for i in tick_idx])

            fig.colorbar(im, ax=ax, label=self._pretty(name))
            fig.tight_layout()
            path = os.path.join(self.result_dir, f"{name}_heatmap.png")
            fig.savefig(path, dpi=200)
            plt.close(fig)
            print(f"[Monitor] Saved {path}")

    # ------------------------------------------------------------------
    # Convenience: generate everything at once
    # ------------------------------------------------------------------
    def finalize(self) -> None:
        """Save all .npy files and generate all plots.

        Call this once at the end of training.
        """
        self.save_npy()
        self.plot_loss_curve()
        self.plot_metric_vs_layers()
        self.plot_metric_vs_epochs()
        self.plot_heatmaps()
        print(f"[Monitor] All results saved to {self.result_dir}/")