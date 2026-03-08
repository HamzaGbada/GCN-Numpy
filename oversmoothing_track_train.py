"""
GCN Oversmoothing Experiment Script

Runs GCN models with increasing depth and measures oversmoothing
using Dirichlet Energy and Rayleigh Quotient.

Outputs:
- saved figures
- saved numpy results
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.datasets import Planetoid

from core.GCN_scratch.model import GCN
from core.utils import GraphUtils
from core.metrics.oversmoothing import OversmoothingMetrics


# -------------------------------------------------------
# Experiment configuration
# -------------------------------------------------------

DATASET = "Cora"

HIDDEN_DIM = 16
EPOCHS = 50
LR = 0.005

# depths to test
# LAYERS_LIST = [1, 2, 4, 8, 16]
LAYERS_LIST = [32]

RESULT_DIR = "results/gcn_oversmoothing"

os.makedirs(RESULT_DIR, exist_ok=True)


# -------------------------------------------------------
# Load dataset
# -------------------------------------------------------

dataset = Planetoid(root="data/Cora", name=DATASET)
data = dataset[0]

N = data.num_nodes

# adjacency
A = np.zeros((N, N))
edge_index = data.edge_index.numpy()

A[edge_index[0], edge_index[1]] = 1
A[edge_index[1], edge_index[0]] = 1

# features
X = data.x.numpy()
X = X / np.linalg.norm(X, axis=1, keepdims=True)
# labels
y = data.y.numpy()
num_labels = len(np.unique(y))
y = np.eye(num_labels)[y]


# -------------------------------------------------------
# Laplacian for oversmoothing metrics
# -------------------------------------------------------

L = OversmoothingMetrics.normalized_laplacian(A)


# -------------------------------------------------------
# Training function
# -------------------------------------------------------

def train_gcn(depth):

    input_dim = X.shape[1]
    output_dim = num_labels

    model = GCN(input_dim, HIDDEN_DIM, output_dim, layers=depth)

    loss_history = []

    dirichlet_history = []
    rayleigh_history = []
    norm_history = []

    for epoch in range(EPOCHS):

        # Forward pass with embeddings
        y_hat, embeddings = model.forward(X, A, return_embeddings=True)

        loss = GraphUtils.loss_function(y, y_hat)
        loss_history.append(loss)

        # ---------------------------
        # Oversmoothing metrics
        # ---------------------------
        de, rq, norm = compute_metrics(embeddings)

        dirichlet_history.append(de)
        rayleigh_history.append(rq)
        norm_history.append(norm)

        # ---------------------------
        # Backprop
        # ---------------------------
        model.backward(y, y_hat, lr=LR)

        print(
            f"[Depth {depth}] Epoch {epoch+1}/{EPOCHS} | "
            f"Loss {loss:.4f} | "
            f"DE_last {de[-1]:.4f}"
        )

    return model, loss_history, dirichlet_history, rayleigh_history, norm_history



# -------------------------------------------------------
# Oversmoothing metrics
# -------------------------------------------------------

def compute_metrics(embeddings):

    de_list = []
    rq_list = []
    norm_list = []

    for H in embeddings:

        de = OversmoothingMetrics.dirichlet_energy(H, L)
        rq = OversmoothingMetrics.rayleigh_quotient(H, L)
        norm = OversmoothingMetrics.feature_norm(H)

        de_list.append(de)
        rq_list.append(rq)
        norm_list.append(norm)

    return de_list, rq_list, norm_list


# -------------------------------------------------------
# Plot metrics
# -------------------------------------------------------

def plot_metrics(depth, de, rq, norm):

    layers = np.arange(len(de))

    plt.figure(figsize=(14,4))

    # Dirichlet Energy
    plt.subplot(1,3,1)
    plt.plot(layers, de, marker="o")
    plt.title(f"Dirichlet Energy (Depth {depth})")
    plt.xlabel("Layer")
    plt.ylabel("DE")

    # Rayleigh Quotient
    plt.subplot(1,3,2)
    plt.plot(layers, rq, marker="o")
    plt.title("Rayleigh Quotient")
    plt.xlabel("Layer")
    plt.ylabel("RQ")

    # Feature norm
    plt.subplot(1,3,3)
    plt.plot(layers, norm, marker="o")
    plt.title("Feature Norm")
    plt.xlabel("Layer")

    plt.tight_layout()

    path = f"{RESULT_DIR}/metrics_depth_{depth}.png"
    plt.savefig(path, dpi=200)

    print("Saved figure:", path)

    plt.close()


def plot_oversmoothing_heatmap(depth, metric_history, metric_name="Dirichlet Energy"):
    """
    metric_history: list of lists -> shape [epochs][layers]
    metric_name: string for plot title
    """
    metric_array = np.array(metric_history)  # shape: [EPOCHS, layers]

    plt.figure(figsize=(10, 6))
    plt.imshow(metric_array, aspect='auto', cmap='viridis')
    plt.colorbar(label=metric_name)
    plt.xlabel("Layer")
    plt.ylabel("Epoch")
    plt.title(f"{metric_name} Heatmap (Depth {depth})")

    path = f"{RESULT_DIR}/{metric_name.replace(' ', '_')}_heatmap_depth_{depth}.png"
    plt.savefig(path, dpi=200)
    print("Saved heatmap:", path)
    plt.close()
# -------------------------------------------------------
# Run experiment
# -------------------------------------------------------

# Run experiment
all_results = {}

for depth in LAYERS_LIST:

    print("\n===================================")
    print(f"Running experiment for depth {depth}")
    print("===================================")

    model, loss_history, de_hist, rq_hist, norm_hist = train_gcn(depth)

    # Plot the final epoch metrics across layers
    de_final = de_hist[-1]
    rq_final = rq_hist[-1]
    norm_final = norm_hist[-1]

    plot_metrics(depth, de_final, rq_final, norm_final)

    # Plot heatmaps over epochs
    plot_oversmoothing_heatmap(depth, de_hist, metric_name="Dirichlet Energy")
    plot_oversmoothing_heatmap(depth, rq_hist, metric_name="Rayleigh Quotient")
    plot_oversmoothing_heatmap(depth, norm_hist, metric_name="Feature Norm")

    # Save results
    all_results[depth] = {
        "loss": loss_history,
        "dirichlet_final": de_final,
        "rayleigh_final": rq_final,
        "norm_final": norm_final,
        "dirichlet_history": de_hist,
        "rayleigh_history": rq_hist,
        "norm_history": norm_hist
    }


# -------------------------------------------------------
# Save results
# -------------------------------------------------------

np.save(f"{RESULT_DIR}/experiment_results.npy", all_results)

print("\nExperiment finished")
print("Results saved in:", RESULT_DIR)