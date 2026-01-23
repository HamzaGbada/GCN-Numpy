# GAT Numpy

## About

This is concise implementation of Graph Attention Network (GAT) for educational purpose using **Numpy**.

## Required theory

### Step 1: Data Representation

* **Adjacency Matrix $A$**:

  * Represents the graph structure where
    $A_{ij} = 1$ if there is an edge between nodes $i$ and $j$, and $0$ otherwise.
  * Self-loops are added:
    $$\tilde{A} = A + I$$

* **Input Feature Matrix $X$**:

  * Represents node features where each row corresponds to a node and each column corresponds to a feature.

### Step 2: Initialization

* **Weight Matrix $W^{(l)}$**:

  * Initialized for each GAT layer $l$.
  * Typically initialized using Xavier/Glorot initialization:
    $$W^{(l)} \sim \mathcal{U}\left(-\sqrt{\frac{6}{F_{in}+F_{out}}}, \sqrt{\frac{6}{F_{in}+F_{out}}}\right)$$

* **Attention Weight Vector $a^{(l)}$**:

  * Learnable vector used to compute attention coefficients.
  * Initialized randomly (Xavier initialization recommended):
    $$a^{(l)} \in \mathbb{R}^{2F_{out}}, \quad
    a^{(l)} \sim \mathcal{U}\left(-\sqrt{\frac{6}{2F_{out}}}, \sqrt{\frac{6}{2F_{out}}}\right)$$

* **Bias Vector $b^{(l)}$** (optional):

  * Initialized to zeros.


### Step 3: Forward Propagation

#### 3.1 Linear Feature Transformation

* Transform input features:
  $$H^{(l)} = X^{(l)} W^{(l)}$$


#### 3.2 Attention Score Computation

* For each edge $(i, j)$ where $\tilde{A}_{ij} = 1$, compute:
  $$e_{ij}^{(l)} =
  \text{LeakyReLU}
  \left(
  {a^{(l)}}^T
  \left[
  h_i^{(l)} || h_j^{(l)}
  \right]
  \right)$$


#### 3.3 Attention Coefficient Normalization

* Normalize attention scores across neighbors:
  $$\alpha_{ij}^{(l)} =
  \frac{\exp(e_{ij}^{(l)})}
  {\sum_{k \in \mathcal{N}(i)} \exp(e_{ik}^{(l)})}$$


#### 3.4 Feature Aggregation

* Aggregate neighbor features:
  $$H^{(l+1)} =
  \sigma
  \left(
  \sum_{j \in \mathcal{N}(i)}
  \alpha_{ij}^{(l)} h_j^{(l)}
  \right)$$

* $\sigma$ denotes a non-linear activation function (e.g., ELU or ReLU).


### Step 4: Loss Calculation

* **Categorical Cross-Entropy Loss** (node classification):
  $$L =
  -\frac{1}{N}
  \sum_{i=1}^{N}
  \sum_{c=1}^{C}
  Y_{ic} \log(\hat{Y}_{ic})$$


### Step 5: Backpropagation

* **Gradient Computation**:

  * Gradients are computed w.r.t:

    * Weight matrices $W^{(l)}$
    * Attention vectors $a^{(l)}$

* **Parameter Update**:
  $$\theta_{new} = \theta_{old} - \alpha \frac{\partial L}{\partial \theta}$$



### Step 6: Training Loop

* Repeat steps 3–5 for a fixed number of epochs or until convergence.



## References

* Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. arXiv. https://arxiv.org/abs/1710.10903  


## Annexe


### **What “WL power” means (quick reminder)**

A GNN has **WL power** if it can distinguish **all graphs that the 1-WL (Weisfeiler–Lehman) test can distinguish**.

The 1-WL update is:

$$
h_i^{(l+1)} =
\text{HASH}\Big(
h_i^{(l)},
;{!{h_j^{(l)} : j \in \mathcal{N}(i)}!}
\Big)
$$

Key property:

> The multiset aggregation is **injective**.

If two neighborhoods differ (in counts or structure), WL will eventually detect it.


### GAT’s update rule

A single-head GAT layer:

$$
h_i^{(l+1)} =
\sum_{j \in \mathcal{N}(i)}
\alpha_{ij}^{(l)} W h_j^{(l)}
$$

where attention weights are:

$$
\alpha_{ij} =\frac{
\exp\big(a^\top [Wh_i \Vert Wh_j]\big)
}{
\sum_{k \in \mathcal{N}(i)}
\exp\big(a^\top [Wh_i \Vert Wh_k]\big)
}
$$

⚠️ **Softmax normalization is the key issue.**


### The core problem: attention = normalized weighted mean

Because of softmax:

$$
\sum_{j \in \mathcal{N}(i)} \alpha_{ij} = 1
$$

So GAT aggregation is:

$$
\boxed{
\text{GAT}(i) = \sum_j \alpha_{ij} x_j
}
$$

This is a **convex combination** of neighbor features.


### Why convex combinations are NOT injective

#### Consider two neighborhoods:

1. Neighborhood A

$$
{x, x, x}
$$

2. Neighborhood B

$$
{x, x}
$$

If attention scores are equal:

$$
\alpha = \frac{1}{|\mathcal{N}(i)|}
$$

Then:

$$
\sum \alpha x = x
$$

➡️ **Different multisets → same output**

❌ Not injective

❌ Not WL-powerful

---

### “But attention weights are learned!”

Yes — **but normalization still kills injectivity**.

Even if attention differs:

$$
\alpha_{ij} =
\frac{e^{s_{ij}}}{\sum_k e^{s_{ik}}}
$$

This enforces:

$$
\alpha_{ij} \in (0,1), \quad \sum_j \alpha_{ij} = 1
$$

So GAT can **reweight**, but it cannot:

* Encode **counts**
* Encode **multiplicity**
* Encode **degree information**


### Formal limitation (Xu et al., 2019)

The GIN paper proves:

> Any GNN whose aggregation is a **weighted mean**
> (including attention with softmax)
>
> **cannot be injective over multisets**

Therefore:

$$
\boxed{
\text{GAT} < \text{1-WL}
}
$$


### Multi-head attention does NOT fix it

Multi-head GAT:

$$
h_i = \Vert_{k=1}^K
\sum_j \alpha_{ij}^{(k)} W^{(k)} h_j
$$

Still:

* Each head is normalized
* Each head loses counts
* Concatenation doesn’t restore injectivity

❌ Still not WL-powerful


### Concrete counterexample (classic)

Two nodes:

#### Graph 1

* Node A has **2 neighbors** with feature `[1]`

#### Graph 2

* Node B has **3 neighbors** with feature `[1]`

For any GAT layer:

$$
h_A = h_B = [1]
$$

WL test **distinguishes** these nodes by degree.
GAT **cannot**.


### Why GIN succeeds where GAT fails

GIN uses:

$$
h_i^{(l+1)} =
\text{MLP}\left(
(1+\varepsilon) h_i + \sum_{j \in \mathcal{N}(i)} h_j
\right)
$$

Key differences:

| Aspect           | GAT           | GIN     |
|------------------|---------------|---------|
| Aggregation      | Weighted mean | **Sum** |
| Normalization    | Yes           | ❌ No    |
| Injective        | ❌             | ✅       |
| Counts preserved | ❌             | ✅       |
| WL power         | ❌             | ✅       |


### Intuition in one sentence

> **Attention tells you *who* matters, but normalization makes you forget *how many* neighbors you have.**

WL needs both.


### Important subtlety

⚠️ GAT is still extremely useful in practice:

* Learns importance
* Handles heterophily
* Works well on real data

But **theoretically**, it cannot match WL power.

---

### Final takeaway

$$
\boxed{
\text{GAT cannot reach WL power because softmax-normalized attention is not an injective multiset function}
}
$$