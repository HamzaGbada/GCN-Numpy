import os
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid

from core import GCN
from core.utils import GraphUtils
from core.metrics.oversmoothing import OversmoothingMetrics


# ---------------------------
# Experiment configuration
# ---------------------------
DATASET = "Cora"
HIDDEN_DIM = 16
EPOCHS = 50
LR = 0.005
LAYERS_LIST = [32]  # you can add 64
RESULT_DIR = "results/gcn_oversmoothing"
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------------------
# Load dataset
# ---------------------------
dataset = Planetoid(root="data/Cora", name=DATASET)
data = dataset[0]
N = data.num_nodes

# features
X = data.x.numpy().astype(np.float64)
X = X / np.linalg.norm(X, axis=1, keepdims=True)
# labels
y = data.y.numpy()
num_labels = len(np.unique(y))
y = np.eye(num_labels)[y]

# adjacency + normalization
A = np.zeros((N, N), dtype=np.float64)
edge_index = data.edge_index.numpy()
A[edge_index[0], edge_index[1]] = 1
A[edge_index[1], edge_index[0]] = 1
I = np.eye(N, dtype=np.float64)
D = np.sum(A + I, axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
A_norm = D_inv_sqrt @ (A + I) @ D_inv_sqrt

# Laplacian for oversmoothing metrics
L = OversmoothingMetrics.normalized_laplacian(A)


# ---------------------------
# Oversmoothing metrics
# ---------------------------
def compute_metrics(embeddings):
    de_list, rq_list, norm_list = [], [], []
    for H in embeddings:
        de_list.append(OversmoothingMetrics.dirichlet_energy(H, L))
        rq_list.append(OversmoothingMetrics.rayleigh_quotient(H, L))
        norm_list.append(OversmoothingMetrics.feature_norm(H))
    return de_list, rq_list, norm_list

# ---------------------------
# Plot metrics
# ---------------------------
def plot_metrics(depth, de, rq, norm):
    layers = np.arange(len(de))
    plt.figure(figsize=(14,4))
    plt.subplot(1,3,1)
    plt.plot(layers, de, marker="o"); plt.title(f"Dirichlet Energy (Depth {depth})"); plt.xlabel("Layer"); plt.ylabel("DE")
    plt.subplot(1,3,2)
    plt.plot(layers, rq, marker="o"); plt.title("Rayleigh Quotient"); plt.xlabel("Layer"); plt.ylabel("RQ")
    plt.subplot(1,3,3)
    plt.plot(layers, norm, marker="o"); plt.title("Feature Norm"); plt.xlabel("Layer")
    plt.tight_layout()
    path = f"{RESULT_DIR}/metrics_depth_{depth}.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved figure:", path)

# ---------------------------
# Training function (compute oversmoothing every 5 epochs)
# ---------------------------
def train_gcn(depth):
    model = GCN(X.shape[1], HIDDEN_DIM, num_labels, layers=depth)
    loss_history, de_history = [], []

    for epoch in range(EPOCHS):
        y_hat, embeddings = model.forward(X, A_norm, return_embeddings=True)
        loss = GraphUtils.loss_function(y, y_hat)
        loss_history.append(loss)

        # compute oversmoothing every 5 epochs
        if epoch % 5 == 0:
            DE_last = OversmoothingMetrics.dirichlet_energy(embeddings[-1], L)
            print(f"[Depth {depth}] Epoch {epoch+1}/{EPOCHS} | Loss {loss:.4f} | DE_last {DE_last:.4f}")
            de_history.append(DE_last)

        model.backward(y, y_hat, lr=LR)

    return model, loss_history, de_history

# ---------------------------
# Run experiment
# ---------------------------
all_results = {}

for depth in LAYERS_LIST:
    print("\n===============================")
    print(f"Running experiment for depth {depth}")
    print("===============================")

    model, loss_history, de_history = train_gcn(depth)
    _, embeddings = model.forward(X, A_norm, return_embeddings=True)
    de, rq, norm = compute_metrics(embeddings)
    plot_metrics(depth, de, rq, norm)

    all_results[depth] = {
        "loss": loss_history,
        "DE_every5": de_history,
        "dirichlet": de,
        "rayleigh": rq,
        "norm": norm
    }

# ---------------------------
# Save results
# ---------------------------
np.save(f"{RESULT_DIR}/experiment_results.npy", all_results)
print("\nExperiment finished")
print("Results saved in:", RESULT_DIR)