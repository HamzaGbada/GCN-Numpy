from src.layers import GCN_Layer, Softmax_Layer
import numpy as np


class GCN_Network:

    def __init__(self, nbr_int, nbr_out, nbr_layer, hidden_sizes, activation):
        self.nbr_int = nbr_int
        self.nbr_out = nbr_out
        self.nbr_layer = nbr_layer
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        self.layers = list()
        # The input layer
        gcn_in = GCN_Layer(nbr_int, hidden_sizes[0], activation)
        self.layers.append(gcn_in)

        # hidden layers
        for l in range(nbr_layer):
            gcn = GCN_Layer(self.layers[-1].W.shape[0], hidden_sizes[l])
            self.layers.append(gcn)

        # output layer
        softmax_output = Softmax_Layer(hidden_sizes[-1], nbr_out)
        self.layers.append(softmax_output)

    def embedding(self, A, X):
        # Loop through all GCN layers
        H = X
        for layer in self.layers[:-1]:
            print(type(layer))
            H = layer.forward(A, H)
        return np.asarray(H)

    def forward(self, A, X):
        # GCN Layers
        H = self.embedding(A, X)

        # Softmax
        p = self.layers[-1].forward(H)

        return np.asarray(p)
