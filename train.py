import numpy as np
from matplotlib import pyplot as plt
from torch_geometric.datasets import Planetoid

from core.GAT_scratch.model import GAT
from core.GCN_scratch.model import GCN
from core.utils import GraphUtils

if __name__ == "__main__":
    dataset = Planetoid(root="data/Cora", name="Cora")

    # Get the data
    data = dataset[0]

    # Extract the adjacency matrix
    adj_matrix = np.zeros((data.num_nodes, data.num_nodes))
    edge_index = data.edge_index.numpy()
    adj_matrix[edge_index[0], edge_index[1]] = 1
    adj_matrix[edge_index[1], edge_index[0]] = 1

    A = adj_matrix
    X = data.x
    y = data.y.numpy()

    # Get the number of unique labels
    num_labels = len(np.unique(data.y))

    # Convert labels to one-hot encoding
    y = np.eye(num_labels)[y]

    input_dim = X.shape[1]
    hidden_dim = 16
    output_dim = num_labels
    epochs = 5
    lr = 0.1

    gcn = GAT(input_dim, hidden_dim, output_dim)

    loss_list = []
    for epoch in range(epochs):
        y_hat = gcn.forward(X, A)

        loss = GraphUtils.loss_function(y, y_hat)
        loss_list.append(loss)
        print(f"the epoch {epoch+1}/{epochs} : \n The current Loss => {loss}")
        gcn.backward(y, y_hat, alpha=lr)
    print("train finished")

    plt.plot(range(epochs), loss_list)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True)  # Add grid lines for better readability
    plt.show()
