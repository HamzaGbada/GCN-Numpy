# GNN from Scratch with NumPy

**GCN · GAT · GIN · GraphSAGE · MPNN**

This project provides **clean, educational, NumPy-only implementations** of major **Graph Neural Network (GNN)** architectures, implemented **from scratch** to expose their mathematical foundations.

The goal is **not performance**, but **understanding**:

* No PyTorch / PyG / DGL for modeling
* Explicit matrix operations
* Clear correspondence between theory and code
* Unified view through the **Message Passing Neural Network (MPNN)** framework


## Implemented Models

* **GCN** — Graph Convolutional Network
* **GAT** — Graph Attention Network
* **GIN** — Graph Isomorphism Network
* **GraphSAGE** — Sample & Aggregate
* **MPNN** — Message Passing Neural Network (general framework)

All models are trained and evaluated on **node classification tasks**.


## Project Structure

```text
.
├── data/                # Graph datasets (raw + processed)
├── docs/                # Theory & math for each model
│   ├── gcn.md
│   ├── gat.md
│   ├── gin.md
│   ├── GraphSAGE.md
│   └── mpnn.md
├── GCN_scratch/         # NumPy implementation
├── GAT_scratch/
├── GIN_scratch/
├── GraphSAGE_scratch/
├── MPNN_scratch/
├── train.py             # Training loop (node classification)
├── pyproject.toml
└── README.md
```

Each `docs/*.md` file contains the **exact mathematical formulation** used by the corresponding NumPy implementation.


## Unifying View: Message Passing Neural Networks (MPNN)

All modern GNNs can be expressed as instances of the **MPNN framework** (Gilmer et al., 2017).

### General MPNN Layer

For each node $i$ at layer $l$:

**Message**
$$m_{ij}^{(l)} = M^{(l)}(h_i^{(l)}, h_j^{(l)}, e_{ij})$$

**Aggregation**
$$m_i^{(l)} = \sum_{j \in \mathcal{N}(i)} m_{ij}^{(l)}$$

**Update**
$$h_i^{(l+1)} = U^{(l)}(h_i^{(l)}, m_i^{(l)})$$

Each GNN defines:

* the **message function** $M$
* the **aggregation operator**
* the **update function** $U$


## Exact Mapping: GCN / GAT / GIN / GraphSAGE → MPNN

### GCN as MPNN

* **Message**
  $$m_{ij} = \frac{1}{\sqrt{d_i d_j}} W h_j$$

* **Aggregation**: normalized sum

* **Update**
  $$h_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i)\cup i} m_{ij}\right)$$

➡ Fixed, non-learnable aggregation (graph Laplacian)


### GAT as MPNN

* **Message**
  $$m_{ij} = \alpha_{ij} W h_j$$

* **Attention**
  $$\alpha_{ij} =
  \text{softmax}_j
  \left(
  \text{LeakyReLU}
  (a^T [Wh_i || Wh_j])
  \right)$$

➡ Learns **edge importance dynamically**


### GIN as MPNN

* **Message**
  $$m_{ij} = h_j$$

* **Aggregation**: **sum (injective)**

* **Update**
  $$h_i^{(l+1)} =
  \text{MLP}
  \left(
  (1+\varepsilon) h_i^{(l)} + \sum_{j \in \mathcal{N}(i)} h_j^{(l)}
  \right)$$

➡ As powerful as the **Weisfeiler–Lehman test**


### GraphSAGE as MPNN

* **Message**
  $$m_{ij} = h_j$$

* **Aggregation**: mean / max / sum

* **Update**

  $$h_i^{(l+1)} = \sigma \left(W \begin{bmatrix} h_i^{(l)} \\ \text{AGG}(\mathcal{N}(i)) \end{bmatrix} \right)$$

➡ Inductive and scalable to unseen nodes


## Comparison Table

| Model         | Message (M)         | Aggregation    | Update (U) | Learnable Aggregation | Laplacian | Attention | MLP      | Inductive | Expressiveness |
|---------------|---------------------|----------------|------------|-----------------------|-----------|-----------|----------|-----------|----------------|
| **GCN**       | $W h_j$             | Normalized sum | Linear + σ | ❌                     | ✅         | ❌         | ❌        | ❌         | Medium         |
| **GAT**       | $\alpha_{ij} W h_j$ | Weighted sum   | Linear + σ | ✅                     | ❌         | ✅         | ❌        | ❌         | High           |
| **GIN**       | $h_j$               | **Sum**        | **MLP**    | ❌                     | ❌         | ❌         | ✅        | ❌         | **Very High**  |
| **GraphSAGE** | $h_j$               | Mean / Max     | Linear + σ | ❌                     | ❌         | ❌         | ❌        | ✅         | Medium         |
| **MPNN**      | Arbitrary           | Any            | Any        | ✅                     | ❌         | Optional  | Optional | Optional  | Maximal        |


## Documentation

| Model     | Theory & Details                         |
|-----------|------------------------------------------|
| WL theorm | [`docs/wl.md`](docs/wl.md)               |
| GCN       | [`docs/gcn.md`](docs/gcn.md)             |
| GAT       | [`docs/gat.md`](docs/gat.md)             |
| GIN       | [`docs/gin.md`](docs/gin.md)             |
| GraphSAGE | [`docs/GraphSAGE.md`](docs/GraphSAGE.md) |
| MPNN      | [`docs/mpnn.md`](docs/mpnn.md)           |

Each document:

* Matches the **NumPy implementation exactly**
* Avoids framework-specific abstractions
* Is suitable for **teaching, labs, and self-study**


## Usage

1. Clone the repository

   ```sh
   git clone https://github.com/HamzaGbada/GCN-Numpy.git
   ```

2. Install required libraries

   ```sh
   uv sync
   ```

3. Activate the virtual environment

   ```sh
   source .venv/bin/activate
   ```

4. Train the model

   ```sh
   python train.py
   ```

## **Note**

We use **[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)** **only for data loading**, not for modeling.

In particular, it is used to load standard graph datasets such as:

* **[Zachary’s Karate Club](http://vlado.fmf.uni-lj.si/pub/networks/data/ucinet/ucidata.htm#zachary)**
* **Cora** (citation network)

All **GNN models and training logic** are implemented **purely in NumPy**.



## References

* Kipf & Welling (2017) — Semi-Supervised GCN
* Veličković et al. (2018) — Graph Attention Networks
* Xu et al. (2019) — Graph Isomorphism Networks
* Hamilton et al. (2017) — GraphSAGE
* Gilmer et al. (2017) — Message Passing Neural Networks
