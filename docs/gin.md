# GIN Numpy

## About

This is concise implementation of Graph Isomorphism Network (GIN) for educational purpose using **Numpy**.

## Required theory 

### Step 1: Data Representation

* **Adjacency Matrix $A$**:

  * Represents the graph structure where
    $A_{ij} = 1$ if there is an edge between nodes $i$ and $j$, and $0$ otherwise.
  * Self-loops are added:
    $$\tilde{A} = A + I$$

* **Input Feature Matrix $X$**:

  * Node feature matrix where each row corresponds to a node and each column corresponds to a feature.
  * Initial node representations:
    $$H^{(0)} = X$$


### Step 2: Initialization

* **MLP Parameters $\text{MLP}^{(l)}$**:

  * Each GIN layer uses a Multi-Layer Perceptron (MLP) instead of a single linear transformation.
  * The MLP parameters (weights and biases) are initialized using standard methods (e.g., Xavier initialization).

* **Epsilon Parameter $\varepsilon^{(l)}$**:

  * A learnable or fixed scalar controlling the importance of the central node:
    $$\varepsilon^{(l)} \in \mathbb{R}$$
  * Common choices:

    * Fixed $\varepsilon = 0$
    * Learnable $\varepsilon^{(l)}$, initialized to 0


### Step 3: Forward Propagation

#### 3.1 Neighborhood Aggregation

* Aggregate features from neighboring nodes using **sum aggregation**:
  $$m_i^{(l)} =
  \sum_{j \in \mathcal{N}(i)}
  h_j^{(l)}$$

* Add the central node contribution:
  $$s_i^{(l)} =
  (1 + \varepsilon^{(l)}) h_i^{(l)} + m_i^{(l)}$$


#### 3.2 Node Update (MLP)

* Update node representations using an MLP:
  $$h_i^{(l+1)} =
  \text{MLP}^{(l)} \left( s_i^{(l)} \right)$$

* In matrix form:
  $$H^{(l+1)} =
  \text{MLP}^{(l)}
  \left(
  (1 + \varepsilon^{(l)}) H^{(l)} + \tilde{A} H^{(l)}
  \right)$$


### Step 4: Loss Calculation

* **Categorical Cross-Entropy Loss** (node classification):
  $$L =
  -\frac{1}{N}
  \sum_{i=1}^{N}
  \sum_{c=1}^{C}
  Y_{ic} \log(\hat{Y}_{ic})$$


### Step 5: Backpropagation

* **Gradient Computation**:

  * Gradients are computed with respect to:

    * MLP weights and biases
    * $\varepsilon^{(l)}$ (if learnable)

* **Parameter Update**:
  $$\theta_{new} = \theta_{old} - \alpha \frac{\partial L}{\partial \theta}$$


### Step 6: Training Loop

* Repeat steps **3–5** for a fixed number of epochs or until convergence.


## References

* Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks? arXiv. https://arxiv.org/abs/1810.00826