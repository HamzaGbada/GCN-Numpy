# GraphSAGE Numpy

## About

This is concise implementation of Graph Sample and Aggregation (GraphSAGE) for educational purpose using **Numpy**.

## Required theory

### Step 1: Data Representation

* **Adjacency Matrix $A$**:

  * Represents the graph structure where
    $A_{ij} = 1$ if there is an edge between nodes $i$ and $j$, and $0$ otherwise.
  * Used to identify neighbor sets:
    $$\mathcal{N}(i) = { j \mid A_{ij} = 1 }$$

* **Input Feature Matrix $X$**:

  * Node feature matrix where each row corresponds to a node and each column corresponds to a feature.
  * Initial node representations:
    $$H^{(0)} = X$$


### Step 2: Initialization

* **Weight Matrices $W^{(l)}$**:

  * Learnable weight matrices for each GraphSAGE layer.
  * Typical GraphSAGE uses **separate weights** for:

    * Neighbor aggregation
    * Self-node transformation
  * Xavier/Glorot initialization is commonly used.

* **Bias Vectors $b^{(l)}$** (optional):

  * Initialized to zero.

* **Aggregator Function**:

  * Fixed, non-learnable function such as:

    * Mean
    * Sum
    * Max
  * (Mean aggregation is most common and simplest for NumPy.)


### Step 3: Forward Propagation

#### 3.1 Neighbor Sampling (Conceptual)

* For each node $i$, sample a subset of neighbors:
  $$\mathcal{S}(i) \subseteq \mathcal{N}(i)$$
* In a full NumPy implementation, all neighbors are typically used.


#### 3.2 Neighbor Aggregation

* Aggregate neighbor representations:
  $$h_{\mathcal{N}(i)}^{(l)} =
  \text{AGGREGATE}
  \left(
  { h_j^{(l)} \mid j \in \mathcal{S}(i) }
  \right)$$

* For **mean aggregation**:
  $$h_{\mathcal{N}(i)}^{(l)} =
  \frac{1}{|\mathcal{S}(i)|}
  \sum_{j \in \mathcal{S}(i)} h_j^{(l)}$$


#### 3.3 Node Update

* Combine self-node and neighbor representations:
  $$h_i^{(l+1)} =
  \sigma
  \left(
  W^{(l)}
\begin{bmatrix}
 h_i^{(l)} \\ h_{\mathcal{N}(i)}^{(l)}
\end{bmatrix}
  b^{(l)}
    \right)$$

* In matrix form:
  $$H^{(l+1)} =
  \sigma
  \left(
  W^{(l)}
  \left[
  H^{(l)} || A H^{(l)}
  \right]
   b^{(l)}
    \right)$$

* $||$ denotes concatenation.


#### 3.4 (Optional) Normalization

* Node embeddings are often normalized:
  $$h_i^{(l+1)} \leftarrow
  \frac{h_i^{(l+1)}}{|h_i^{(l+1)}|_2}$$


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

    * Weight matrices $W^{(l)}$
    * Bias vectors $b^{(l)}$

* **Parameter Update**:
  $$\theta_{new} = \theta_{old} - \alpha \frac{\partial L}{\partial \theta}$$


### Step 6: Training Loop

* Repeat steps **3–5** for a fixed number of epochs or until convergence.

## References

* Hamilton, W. L., Ying, R., & Leskovec, J. (2018). Inductive representation learning on large graphs. arXiv. https://arxiv.org/abs/1706.02216