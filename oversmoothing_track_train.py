"""
Oversmoothing Experiment Runner — Model-Agnostic Deep GNN Pipeline.

Trains a configurable GNN model on the Cora dataset and records per-layer
oversmoothing metrics (Dirichlet Energy, Rayleigh Quotient, Pairwise
Distance, Feature Variance) at regular intervals.  All results (plots and
.npy arrays) are saved automatically via ``OversmoothingMonitor``.

Usage:
    python oversmoothing_track_train.py
"""

import os
import numpy as np
from torch_geometric.datasets import Planetoid

from core import get_model
from core.utils import GraphUtils
from core.monitoring.ovesmoothing_monitor import OversmoothingMonitor

# ===================================================================
# Experiment Configuration  (edit these constants to run experiments)
# ===================================================================
MODEL_NAME = "GCN"          # One of: "GCN", "GAT", "GIN", "GraphSAGE"
HIDDEN_DIM = 8             # Hidden feature dimension
EPOCHS = 50                 # Total training epochs
LR = 0.005                  # Learning rate
LAYERS_LIST = [4, 8, 16, 32]          # Depths to evaluate (e.g. [4, 8, 16, 32, 64])
METRIC_INTERVAL = 5         # Compute oversmoothing metrics every N epochs
RESULT_DIR = "results/gcn_oversmoothing"

# ===================================================================
# Data Loading — Cora (float64 throughout)
# ===================================================================
dataset = Planetoid(root="data/Cora", name="Cora")
data = dataset[0]
N = data.num_nodes

# Node features — float64, row-normalised
X = data.x.numpy().astype(np.float64)
row_norms = np.linalg.norm(X, axis=1, keepdims=True)
row_norms = np.where(row_norms < 1e-12, 1.0, row_norms)  # guard zero rows
X = X / row_norms

# Labels — one-hot encoded
y = data.y.numpy()
num_labels = len(np.unique(y))
y = np.eye(num_labels, dtype=np.float64)[y]

# Raw adjacency matrix (symmetric)
A = np.zeros((N, N), dtype=np.float64)
edge_index = data.edge_index.numpy()
A[edge_index[0], edge_index[1]] = 1
A[edge_index[1], edge_index[0]] = 1

# Normalised adjacency: D̃^{-½} Ã D̃^{-½}  where Ã = A + I
I = np.eye(N, dtype=np.float64)
D_tilde = np.sum(A + I, axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(D_tilde))
A_norm = D_inv_sqrt @ (A + I) @ D_inv_sqrt


# ===================================================================
# Training Function
# ===================================================================
def train_model(depth: int) -> None:
    """Train a single model at the given depth and record metrics.

    Args:
        depth: Number of hidden layers.
    """
    result_dir = os.path.join(RESULT_DIR, f"depth_{depth}")
    os.makedirs(result_dir, exist_ok=True)

    # --- Instantiate model ---
    model = get_model(MODEL_NAME, X.shape[1], HIDDEN_DIM, num_labels, layers=depth)

    # --- Instantiate monitor ---
    monitor = OversmoothingMonitor(A, result_dir=result_dir,
                                   metric_interval=METRIC_INTERVAL)

    # --- Training loop ---
    for epoch in range(EPOCHS):
        # Forward pass with embeddings
        result = model.forward(X, A_norm, return_embeddings=True)
        if isinstance(result, tuple):
            y_hat, embeddings = result
        else:
            y_hat = result
            embeddings = []

        # Compute loss
        loss = GraphUtils.loss_function(y, y_hat)
        monitor.record_loss(loss)

        # Record oversmoothing metrics at interval
        if monitor.should_record(epoch) and len(embeddings) > 0:
            monitor.record_metrics(epoch, embeddings)
            de_last = embeddings[-1]
            from core.metrics.oversmoothing import OversmoothingMetrics
            L = OversmoothingMetrics.normalized_laplacian(A)
            de_val = OversmoothingMetrics.dirichlet_energy(de_last, L)
            print(
                f"[{MODEL_NAME} depth={depth}] "
                f"Epoch {epoch + 1:3d}/{EPOCHS} | "
                f"Loss {loss:.6f} | "
                f"DE(last) {de_val:.6f}"
            )

        # Backward pass
        model.backward(y, y_hat, lr=LR)

    # --- Finalise: save .npy + generate all plots ---
    monitor.finalize()


# ===================================================================
# Run Experiments
# ===================================================================
if __name__ == "__main__":
    for depth in LAYERS_LIST:
        print("\n" + "=" * 60)
        print(f"  Experiment: {MODEL_NAME}  |  depth = {depth}  |  epochs = {EPOCHS}")
        print("=" * 60)
        train_model(depth)

    print("\n✓ All experiments finished.")
    print(f"  Results saved under: {RESULT_DIR}/")