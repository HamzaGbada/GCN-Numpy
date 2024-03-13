import numpy as np
from dgl.data import CoraGraphDataset
from torch_geometric.datasets import KarateClub, Planetoid

from GCN_scratch.model import GCN
from GCN_scratch.utils import GraphUtils

if __name__ == "__main__":
    dataset = Planetoid(root='data/Cora', name='Cora')

    # Get the data
    data = dataset[0]

    # Extract the adjacency matrix
    adj_matrix = np.zeros((data.num_nodes, data.num_nodes))
    edge_index = data.edge_index.numpy()
    adj_matrix[edge_index[0], edge_index[1]] = 1
    adj_matrix[edge_index[1], edge_index[0]] = 1

    # Convert to a sparse matrix
    A = adj_matrix
    print(f"the adjancy matrix {A}")
    X = data.x
    print(f"the node feature {X.shape}, the number of node {data.num_nodes}")
    y = data.y.numpy()

    # Get the number of unique labels
    num_labels = len(np.unique(data.y))

    # Convert labels to one-hot encoding
    y = np.eye(num_labels)[y]
    print(f"the node label {y}, the node label shape {y.shape},the number of node {data.num_nodes}")

    input_dim = X.shape[1]
    hidden_dim = 16  # Choose the size of the hidden layer
    output_dim = np.unique(y).shape[0]
    epochs = 10
    lr = 0.1

    gcn = GCN(input_dim, hidden_dim, output_dim)

    loss_list = []
    for epoch in range(epochs):
        y_hat = gcn.forward(X, A)

        loss = GraphUtils.loss_function(y, y_hat)
        loss_list.append(loss)
        print(f"the epoch {epoch+1}/{epochs} : \n The current Loss => {loss}")
        gcn.backward(y, y_hat, alpha=lr)
    print("train finished")