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


Below is a **fully detailed, math-level explanation** of **Step 4 (Loss)** and **Step 5 (Backpropagation)** for **GIN**, with **all intermediate derivatives**, written so you can **directly translate it to NumPy code**.

I will stay consistent with your notation and **explicitly show where gradients flow**.

### Step 4 — Loss Calculation (Detailed)

We assume **node classification**.



* **4.1 Model output**

Final GIN layer output (logits):

$$
Z = H^{(L)} \in \mathbb{R}^{N \times C}
$$

Apply **softmax** row-wise:

$$
\hat{Y}*{ic} =
\frac{e^{Z*{ic}}}
{\sum_{k=1}^{C} e^{Z_{ik}}}
$$


* **4.2 One-hot ground truth**

$$
Y \in \mathbb{R}^{N \times C}, \quad
Y_{ic} \in {0,1}
$$


* **4.3 Cross-entropy loss (mean over nodes)**

$$
L =
-\frac{1}{N}
\sum_{i=1}^{N}
\sum_{c=1}^{C}
Y_{ic} \log(\hat{Y}_{ic})
$$


* **4.4 Gradient of loss w.r.t logits (CRITICAL RESULT)**

For **softmax + cross-entropy**, the derivative simplifies to:

$$
\boxed{
\frac{\partial L}{\partial Z}
\frac{1}{N} (\hat{Y} - Y)
}
$$

This is why most implementations **never compute softmax gradients explicitly**.


### Step 5 — Backpropagation (Full Derivation)

We now backpropagate through **one GIN layer**.


* **Recap: Forward equations of one GIN layer**

$$
\begin{aligned}
M^{(l)} &= \tilde{A} H^{(l)} \\ 
S^{(l)} &= (1 + \varepsilon^{(l)}) H^{(l)} + M^{(l)} \\
H^{(l+1)} &= \text{MLP}^{(l)}(S^{(l)})
\end{aligned}
$$

* **Structure of the MLP**

We assume a **2-layer MLP**:

$$
\begin{aligned}
Z_1 &= S W_1 + b_1 \\
A_1 &= \text{ReLU}(Z_1) \\
Z_2 &= A_1 W_2 + b_2 \\
H^{(l+1)} &= Z_2
\end{aligned}
$$

* **5.1 Backprop from loss to last GIN layer**

From Step 4:

$$
\delta^{(l+1)} =
\frac{\partial L}{\partial H^{(l+1)}}
\frac{1}{N} (\hat{Y} - Y)
$$

This is the **error signal entering the last GIN layer**.

* **5.2 Gradients inside the MLP**


* **Gradient w.r.t second linear layer**

$$
\frac{\partial L}{\partial W_2}=
A_1^T \delta^{(l+1)}
$$

$$
\frac{\partial L}{\partial b_2}=\sum_{i=1}^{N} \delta^{(l+1)}_i
$$

* **Backprop through ReLU**

$$
\delta_1 =
\delta^{(l+1)} W_2^T
\odot \mathbb{1}(Z_1 > 0)
$$

* **Gradient w.r.t first linear layer**

$$
\frac{\partial L}{\partial W_1}= S^T \delta_1
$$

$$
\frac{\partial L}{\partial b_1}= \sum_{i=1}^{N} \delta_1^i
$$


* **5.3 Gradient w.r.t aggregated features $S^{(l)}$**

$$
\boxed{
\frac{\partial L}{\partial S^{(l)}}
=
\delta_1 W_1^T
}
$$

This gradient now flows **outside the MLP**.


* **5.4 Gradient w.r.t epsilon $\varepsilon^{(l)}$**

Recall:

$$
S^{(l)} =
(1 + \varepsilon^{(l)}) H^{(l)} + \tilde{A} H^{(l)}
$$

Derivative:

$$
\frac{\partial S^{(l)}}{\partial \varepsilon^{(l)}}
=
H^{(l)}
$$

Chain rule:

$$
\boxed{
\frac{\partial L}{\partial \varepsilon^{(l)}}
=
\sum_{i=1}^{N}
\left(
\frac{\partial L}{\partial S^{(l)}_i}
\cdot
H^{(l)}_i
\right)
}
$$

📌 Scalar result (one ε per layer)

* **5.5 Gradient w.r.t previous layer embeddings $H^{(l)}$**

From:

$$ S^{(l)} =
(1 + \varepsilon^{(l)}) H^{(l)} + \tilde{A} H^{(l)}$$

Derivative:

$$
\frac{\partial S^{(l)}}{\partial H^{(l)}} = (1 + \varepsilon^{(l)}) I + \tilde{A}
$$

So:

$$
\boxed{
\frac{\partial L}{\partial H^{(l)}}
=
\left(
(1 + \varepsilon^{(l)}) I + \tilde{A}^T
\right)
\frac{\partial L}{\partial S^{(l)}}
}
$$

(For undirected graphs, $\tilde{A}^T = \tilde{A}$)


* **5.6 Parameter update rule (all parameters)**

For **every parameter** $\theta$:

$$
\theta \leftarrow
\theta - \alpha \frac{\partial L}{\partial \theta}
$$

Specifically:

* $W_1, b_1$
* $W_2, b_2$
* $\varepsilon^{(l)}$
* (Repeated for all layers)

*  **Full Backprop Flow (visual summary)**

```
Loss
 ↓
Softmax
 ↓
H(L)
 ↓
MLP(L)
 ↓
ε(L)
 ↓
Aggregation
 ↓
H(L−1)
 ↓
...
```

Everything is **trained jointly**.


✔ No normalization anywhere in GIN



### Step 6: Training Loop

* Repeat steps **3–5** for a fixed number of epochs or until convergence.


## References

* Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks? arXiv. https://arxiv.org/abs/1810.00826

## Annexe


### What does *injective* mean in GIN context?

#### General definition (math)

A function ( f ) is **injective** if:

$$
f(x_1) = f(x_2) \Rightarrow x_1 = x_2
$$

➡️ Different inputs **must** produce different outputs.

---

##### Injectivity in GNNs

In GNNs, the function we care about is:

$$
\text{AGGREGATE}(\text{neighbor features})
$$

Injectivity here means:

> **Two different node neighborhoods must not collapse to the same embedding.**

Formally:

$$
\text{AGG}(\mathcal{N}_1) = \text{AGG}(\mathcal{N}_2)
\Rightarrow
\mathcal{N}_1 = \mathcal{N}_2
$$

---

##### Why this matters

If aggregation is **not injective**, then:

* Different graph structures
* Different node neighborhoods

can become **indistinguishable** to the model.

➡️ Once collapsed, **no deeper layer can recover the difference**.

---

### Why normalization breaks injectivity

Let’s look at **mean / normalized aggregation**, used in GCN:

$$
h_i^{(l+1)} =
\sum_{j \in \mathcal{N}(i)}
\frac{1}{\sqrt{d_i d_j}}
W h_j^{(l)}
$$

or mean aggregation:

$$
\text{mean}({x_1, x_2, x_3})
$$



### Example (very important)

Consider two neighborhoods:

### Neighborhood A

$$
{1, 1, 1}
$$

### Neighborhood B

$$
{1, 1}
$$

Mean aggregation:

$$
\text{mean}(A) = 1
\quad
\text{mean}(B) = 1
$$

➡️ **Different multisets → same embedding**

🚨 **Not injective**


### Degree normalization also causes collapse

Let:

$$
\tilde{A} = D^{-1/2} A D^{-1/2}
$$

Two different neighborhoods with different degrees can be **scaled into the same vector**.

#### Intuition:

* Degree information is **divided out**
* Multiplicity of neighbors is **lost**


### Why sum aggregation is injective (in theory)

GIN uses:

$$
\sum_{j \in \mathcal{N}(i)} h_j
$$

Sum preserves:

* Counts
* Multiplicity
* Structure

The paper proves:

> **Sum aggregation + injective MLP = injective neighborhood function**


### Why GCN / GraphSAGE are weaker than GIN

| Model     | Aggregation           | Injective?            |
| --------- | --------------------- | --------------------- |
| GCN       | Mean / normalized sum | ❌ No                  |
| GraphSAGE | Mean / max            | ❌ No                  |
| GAT       | Weighted mean         | ❌ No                  |
| **GIN**   | **Sum + MLP**         | ✅ Yes (theoretically) |

⚠️ GAT **learns attention**, but still normalizes with softmax → loses multiplicity.


### What does *isomorphism* mean here?

#### Graph isomorphism (GI problem)

Two graphs $G_1$ and $G_2$ are **isomorphic** if:

* They are identical **up to node relabeling**
* Structure is the same

Example:

```
1—2—3     a—b—c
```

Same graph, different labels.


#### Weisfeiler–Lehman (WL) test

The **WL test**:

* Iteratively aggregates neighbor labels
* Checks whether graphs become distinguishable

It is a **powerful heuristic** for graph isomorphism.


#### Key result of the GIN paper

> **GIN is as powerful as the 1-WL test**

Meaning:

* If WL can distinguish two graphs
* GIN **can also** distinguish them (with enough capacity)


### Why GIN is called *Graph Isomorphism Network*

Because:

1. It uses **injective aggregation**
2. It preserves neighborhood structure
3. It matches the discriminative power of **Weisfeiler–Lehman**
4. It avoids normalization that destroys structural information

So:

$$
\boxed{
\text{GIN} \approx \text{Neural version of WL test}
}
$$


### Intuition in one sentence

> **GIN is designed so that different graph structures never collapse to the same representation unless they are truly identical (isomorphic).**


### Visual summary

```
GCN / GAT / SAGE
    ↓
Normalize / Average
    ↓
Lose counts → Not injective → Structure collapse

GIN
    ↓
Sum + MLP
    ↓
Injective → Structure preserved → WL-power
```

### Final takeaway

| Concept       | Meaning                                              |
| ------------- | ---------------------------------------------------- |
| Injective     | No two different neighborhoods map to same embedding |
| Normalization | Destroys multiplicity information                    |
| GIN           | Sum + MLP ensures injectivity                        |
| Isomorphism   | Same graph up to relabeling                          |
| GIN name      | Matches WL graph isomorphism test                    |

---

If you want, next I can:

* 🔹 Prove injectivity formally (paper-level)
* 🔹 Show counterexamples for GCN/GAT
* 🔹 Explain why **ε** is crucial
* 🔹 Compare **GIN vs MPNN expressiveness**

Just tell me 👍
