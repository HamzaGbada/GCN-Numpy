import numpy as np
from dgl.data import CoraGraphDataset

from GCN_scratch.model import GCN
from GCN_scratch.utils import GraphUtils

if __name__ == "__main__":
    dataset = CoraGraphDataset()
    g = dataset[0]
    A = g.adjacency_matrix()
    X = g.ndata["feat"]
    y = g.ndata['label']

    input_dim = X.shape[1]
    hidden_dim = 16  # Choose the size of the hidden layer
    output_dim = y.shape[1]
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