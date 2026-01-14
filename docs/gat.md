# GAT Numpy

## About

This is concise implementation of Graph Attention Network (GAT) for educational purpose using **Numpy**.


## Required theory

### Step 1: Data Representation

* **Adjacency Matrix $A$**:

  * Represents the graph structure where
    $A_{ij} = 1$ if there is an edge between nodes $i$ and $j$, and $0$ otherwise.
  * Self-loops are added:
    $$
    \tilde{A} = A + I
    $$

* **Input Feature Matrix $X$**:

  * Represents node features where each row corresponds to a node and each column corresponds to a feature.

---

### Step 2: Initialization

* **Weight Matrix $W^{(l)}$**:

  * Initialized for each GAT layer $l$.
  * Typically initialized using Xavier/Glorot initialization:
    $$
    W^{(l)} \sim \mathcal{U}\left(-\sqrt{\frac{6}{F_{in}+F_{out}}}, \sqrt{\frac{6}{F_{in}+F_{out}}}\right)
    $$

* **Attention Weight Vector $a^{(l)}$**:

  * Learnable vector used to compute attention coefficients.
  * Initialized randomly (Xavier initialization recommended):
    $$
    a^{(l)} \in \mathbb{R}^{2F_{out}}, \quad
    a^{(l)} \sim \mathcal{U}\left(-\sqrt{\frac{6}{2F_{out}}}, \sqrt{\frac{6}{2F_{out}}}\right)
    $$

* **Bias Vector $b^{(l)}$** (optional):

  * Initialized to zeros.

---

### Step 3: Forward Propagation

#### 3.1 Linear Feature Transformation

* Transform input features:
  $$
  H^{(l)} = X^{(l)} W^{(l)}
  $$

---

#### 3.2 Attention Score Computation

* For each edge ( (i, j) ) where ( \tilde{A}*{ij} = 1 ), compute:
  $$
  e_{ij}^{(l)} =
  \text{LeakyReLU}
  \left(
  {a^{(l)}}^T
  \left[
  h_i^{(l)} , || , h_j^{(l)}
  \right]
  \right)
  $$

---

#### 3.3 Attention Coefficient Normalization

* Normalize attention scores across neighbors:
  $$
  \alpha_{ij}^{(l)} =
  \frac{\exp(e_{ij}^{(l)})}
  {\sum_{k \in \mathcal{N}(i)} \exp(e_{ik}^{(l)})}
  $$

---

#### 3.4 Feature Aggregation

* Aggregate neighbor features:
  $$
  H^{(l+1)} =
  \sigma
  \left(
  \sum_{j \in \mathcal{N}(i)}
  \alpha_{ij}^{(l)} h_j^{(l)}
  \right)
  $$

* $\sigma$ denotes a non-linear activation function (e.g., ELU or ReLU).

---

### Step 4: Loss Calculation

* **Categorical Cross-Entropy Loss** (node classification):
  $$
  L =
  -\frac{1}{N}
  \sum_{i=1}^{N}
  \sum_{c=1}^{C}
  Y_{ic} \log(\hat{Y}_{ic})
  $$

---

### Step 5: Backpropagation

* **Gradient Computation**:

  * Gradients are computed w.r.t:

    * Weight matrices $W^{(l)}$
    * Attention vectors $a^{(l)}$

* **Parameter Update**:
  $$
  \theta_{new} = \theta_{old} - \alpha \frac{\partial L}{\partial \theta}
  $$

---

### Step 6: Training Loop

* Repeat steps 3–5 for a fixed number of epochs or until convergence.




## References

* Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. arXiv. https://arxiv.org/abs/1710.10903  


