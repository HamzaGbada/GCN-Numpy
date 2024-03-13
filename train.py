import numpy as np

from GCN_scratch.model import GCN
from GCN_scratch.utils import GraphUtils

X = np.random.randint(0,10, (10,10))
A = np.random.randint(0,10, (10,10))
y = np.random.randint(0,10, (10,10))
input_dim = X.shape[1]
hidden_dim = 16  # Choose the size of the hidden layer
output_dim = y.shape[1]
lr = 0.1

gcn = GCN(input_dim, hidden_dim, output_dim)

for epoch in range(10):
    y_hat = gcn.forward(X, A)
    loss = GraphUtils.loss_function(y, y_hat)
    gcn.backward(y, y_hat,)
