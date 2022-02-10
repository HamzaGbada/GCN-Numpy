import numpy as np
import networkx as nx
import scipy.linalg as la
import matplotlib.pyplot as plt


from src.NeuralNetwork import GCN_Network
from src.layers import Utils

# Here I used networkx Binomial graph as dataset (you can changed it or test it with other params) it params are
# choosing randomely
# https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.binomial_graph.html
# you are free to change the data I already test it on Zachary’s Karate Club graph, and The Turan Graph
# For more check
# https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html
# https://networkx.org/documentation/stable/reference/generated/networkx.generators.classic.turan_graph.html#
graph = nx.binomial_graph(100,0.6)

# Graph plotting
nx.draw_spectral(graph)
plt.show

# Adjacent Matrix
A = np.array(nx.to_numpy_matrix(graph))

# To create Laplace Matrix
degree_of_nodes = np.count_nonzero(A, axis=1)
D = np.diag(degree_of_nodes)
D_inv_sqrt = np.linalg.inv(la.sqrtm(D))

Id = np.identity(graph.number_of_nodes())

A_Laplacien = Id - np.dot(D_inv_sqrt, np.dot(A, D_inv_sqrt))

# input feature (but we don't have any node feature so we assign a vector contains one for each node)
X = np.identity(graph.number_of_nodes())

# Class number and labels
from networkx.algorithms.community import greedy_modularity_communities

# to regroup node using community algorithm
communities = greedy_modularity_communities(graph)
colors = np.zeros(graph.number_of_nodes())
for i, com in enumerate(communities):
    colors[list(com)] = i

nbr_class = len(communities)
labels = np.eye(nbr_class)[colors.astype(int)]

# GCN model creation
gcn_model = GCN_Network(
    nbr_int=graph.number_of_nodes(),
    nbr_out=nbr_class,
    nbr_layer=2,
    hidden_sizes=[16, 2],
    activation=np.tanh
)

y_pred = gcn_model.forward(A_Laplacien, X)
embed = gcn_model.embedding(A_Laplacien, X)
Utils.xent(y_pred, labels).mean()