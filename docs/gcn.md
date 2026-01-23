# GCN Numpy

## About

This is concise implementation of Graph Convolution Network (GCN) for educational purpose using **Numpy**.

## Required theory


### Step 1: Data Representation
- **Adjacency Matrix $A$**:
  - Represents the graph structure where $A_{ij} = 1$ if there is an edge between nodes $i$ and $j$, and $A_{ij} = 0$ otherwise.
- **Input Feature Matrix $X$**:
  - Represents node features where each row corresponds to a node and each column corresponds to a feature.

### Step 2: Initialization
- **Weight Matrices $W^{(l)}$**:
  - Initialize weight matrices for each layer $l$ of the GCN.
- **Bias Vectors $b^{(l)}$** (optional):
  - Optionally, initialize bias vectors for each layer.

### Step 3: Forward Propagation
- **Normalized Graph Laplacian $\tilde{L}$**:
  - Compute the normalized graph Laplacian: 
    - $$\tilde{L} = I - D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$$
- **Graph Convolution Operation**:
  - Compute the node representation matrix at layer $l+1$:
    - $$H^{(l+1)} = \sigma(\tilde{L} H^{(l)} W^{(l)})$$
  - $\sigma$ is the activation function.

### Step 4: Loss Calculation
- **Loss Function**:
  - Compute the loss function for node classification (which is our case, we used Categorical Cross-Entropy):
    - $$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} Y_{ij} \log(\hat{Y}_{ij})$$

### Step 5: Backpropagation
- **Gradient Computation**:
  - Compute the gradients of the loss function with respect to the model parameters using backpropagation.
  - Example: $\frac{\partial L}{\partial W} = \frac{1}{N} (X^T A^T) (\hat{Y} - Y)$.
- **Parameter Update**:
  - Update the model parameters using gradient descent or another optimization algorithm.
    - $$W_{new} = W_{old} - \alpha \frac{\partial L}{\partial W}$$

### Step 6: Training Loop
- **Iteration**:
  - Repeat steps 3-5 iteratively until convergence or for a fixed number of epochs.


## References

* Kipf, Thomas & Welling, Max. (2016). Semi-Supervised Classification with Graph Convolutional Networks.  


