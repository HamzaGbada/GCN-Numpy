# MPNN Numpy

## About

This is concise implementation of Message Passing Neural Network (MPNN) for educational purpose using **Numpy**.


## Required theory

### Step 1: Data Representation

* **Adjacency Matrix $A$**:

  * Represents the graph structure where
    $A_{ij} = 1$ if there is an edge between nodes $i$ and $j$, and $0$ otherwise.
  * Defines the neighborhood:
    $$\mathcal{N}(i) = { j \mid A_{ij} = 1 }$$

* **Input Feature Matrix $X$**:

  * Node feature matrix where each row corresponds to a node and each column corresponds to a feature.
  * Initial node states:
    $$H^{(0)} = X$$

* **(Optional) Edge Feature Matrix $E$**:

  * Represents edge attributes (if available), indexed by $(i, j)$.

### Step 2: Initialization

* **Message Function Parameters $\theta_M^{(l)}$**:

  * Parameters of the message function $M^{(l)}$.
  * Typically implemented as a linear layer or small MLP.
  * Initialized using Xavier/Glorot initialization.

* **Update Function Parameters $\theta_U^{(l)}$**:

  * Parameters of the node update function $U^{(l)}$.
  * Can be a linear layer, MLP, or gated unit (e.g., GRU).

* **Bias Vectors** (optional):

  * Initialized to zero.


### Step 3: Forward Propagation

#### 3.1 Message Computation

* For each node $i$ and neighbor $j$, compute messages:
  $$m_{ij}^{(l)} =
  M^{(l)}
  \left(
  h_i^{(l)}, h_j^{(l)}, e_{ij}
  \right)$$

* In a simple NumPy implementation (no edge features):
  $$m_{ij}^{(l)} = W_M^{(l)} h_j^{(l)}$$


#### 3.2 Message Aggregation

* Aggregate incoming messages:
  $$m_i^{(l)} =
  \sum_{j \in \mathcal{N}(i)} m_{ij}^{(l)}$$

* Other aggregators (mean, max) are also possible.


#### 3.3 Node State Update

* Update node representations:
  $$h_i^{(l+1)} =
  U^{(l)}
  \left(
  h_i^{(l)}, m_i^{(l)}
  \right)$$

* Simple update function:
  $$h_i^{(l+1)} =
  \sigma
  \left(
  W_U^{(l)}
  \begin{bmatrix}
  h_i^{(l)} \\
  m_i^{(l)}
  \end{bmatrix}
b^{(l)}
    \right)$$

* In matrix form:
  $$H^{(l+1)} =
  \sigma
  \left(
  W_U^{(l)}
  \left[
  H^{(l)} || A H^{(l)}
  \right]
   b^{(l)}
    \right)$$


### Step 4: Readout (Optional, Graph-Level Tasks)

* For graph-level prediction, aggregate node embeddings:
  $$h_G = \text{READOUT}({ h_i^{(L)} })$$
* Common readout functions: sum, mean, max.


### Step 5: Loss Calculation

* **Categorical Cross-Entropy Loss** (node classification):
  $$L =
  -\frac{1}{N}
  \sum_{i=1}^{N}
  \sum_{c=1}^{C}
  Y_{ic} \log(\hat{Y}_{ic})$$


### Step 6: Backpropagation

* **Gradient Computation**:

  * Gradients are computed with respect to:

    * Message function parameters $\theta_M^{(l)}$
    * Update function parameters $\theta_U^{(l)}$

* **Parameter Update**:
  $$\theta_{new} = \theta_{old} - \alpha \frac{\partial L}{\partial \theta}$$


### Step 7: Training Loop

* Repeat steps **3–6** for a fixed number of epochs or until convergence.

## References

* Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural message passing for quantum chemistry. arXiv. https://arxiv.org/abs/1704.01212