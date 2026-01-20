# Weisfeiler–Lehman Explained

## 1 What is the Weisfeiler–Lehman (1-WL) test?

The **1-WL test** (also called *color refinement*) is an **algorithmic test** to check whether two graphs are **non-isomorphic**.

Important:

* If WL says *different* → graphs are **definitely non-isomorphic**
* If WL says *same* → graphs **may or may not** be isomorphic

So WL is **sound but not complete**.
(isomorphic graph where $\exists$ a bijection between them (not really maybe injection) like A-B-C and 1-2-3)

## How 1-WL works (algorithmically)

Each node has a **color (label)**.

### Initialization (iteration 0)

All nodes get the same color (or initial features):

```
Iteration 0:
All nodes → color A
```

---

### Iteration t → t+1

For each node:

1. Collect **multiset of neighbor colors**
2. Combine with its own color
3. Hash the result into a new color

Formally:
$$
c_i^{(t+1)} =
\text{HASH}\Big(
c_i^{(t)},
;{!{ c_j^{(t)} : j \in \mathcal{N}(i) }!}
\Big)
$$

---

### Stop condition

* Colors stabilize (no change), or
* A difference between graphs appears

---

##  Concrete Example 1 — WL succeeds

### Graph A: Path of 3 nodes

```
1 — 2 — 3
```

### Graph B: Triangle

```
1 — 2
 \  /
  3
```

---

### Iteration 0

All nodes have same color:

```
A A A
```

---

### Iteration 1

**Graph A**

* Node 2: neighbors = {A, A}
* Nodes 1 & 3: neighbors = {A}

Colors become:

```
B — C — B
```

**Graph B**

* All nodes have neighbors = {A, A}

Colors become:

```
D D D
```

🚨 Different color distributions → **Graphs are non-isomorphic**

✔ WL succeeds

---

## Concrete Example 2 — WL detects degree differences

### Graph A

Node with 2 neighbors

### Graph B

Node with 3 neighbors

At iteration 1:

* Neighbor multisets differ in size
* Hash differs

✔ WL distinguishes them
❌ GCN / GAT cannot (normalization)

---

## Concrete Example 3 — WL fails (classic failure)

### Two non-isomorphic regular graphs

Both graphs:

* All nodes degree = 3
* Same number of nodes
* Same local structure

Example:

* Petersen graph
* Certain strongly regular graphs

---

### Iteration 0

All nodes same color

### Iteration 1

All nodes see same multiset `{A, A, A}`

### Iteration t

Colors **never split**

🚨 WL says graphs are identical
❌ They are actually non-isomorphic

➡️ This is a **known limitation of 1-WL**

---

## What the WL Theorem actually says (important)

### Informal statement (Xu et al., Morris et al.)

> Any message-passing GNN is **at most as powerful as the 1-WL test**.

Meaning:

* If 1-WL cannot distinguish two graphs
* **No standard GNN can**, regardless of depth

---

### Stronger statement

> **<p style="color: red;"> A GNN is WL-powerful if and only if its aggregation function is injective over multisets.</p>**

This is where **GIN** comes from.

---

## Mapping WL to GNN layers

| WL step                 | GNN equivalent                |
| ----------------------- | ----------------------------- |
| Collect neighbor colors | Aggregate neighbor embeddings |
| Hash                    | MLP                           |
| Color                   | Node embedding                |

So:
$$
\text{WL} \approx \text{GNN with injective aggregation}
$$

---

## Why GIN matches WL exactly

GIN update:
$$
h_i^{(l+1)} =
\text{MLP}\left(
(1+\varepsilon) h_i^{(l)} +
\sum_{j \in \mathcal{N}(i)} h_j^{(l)}
\right)
$$

* Sum = injective over multisets
* MLP = universal approximator
* ε = separates self from neighbors

✔ Matches WL power
✔ Cannot exceed WL power

---

## Why GCN / GAT / SAGE fail WL

| Model     | Aggregation         | Problem            |
|-----------|---------------------|--------------------|
| GCN       | Mean / norm         | Loses counts       |
| GraphSAGE | Mean / max          | Not injective      |
| GAT       | Attention + softmax | Convex combination |

Example:

```
{A, A} and {A, A, A}
→ same mean
```

WL distinguishes, these models do not.

---

## Case Summary Table

| Case                      | WL | GCN | GAT | GIN |
| ------------------------- | -- | --- | --- | --- |
| Different degrees         | ✔  | ❌   | ❌   | ✔   |
| Different neighbor counts | ✔  | ❌   | ❌   | ✔   |
| Regular graphs            | ❌  | ❌   | ❌   | ❌   |
| WL-equivalent graphs      | ❌  | ❌   | ❌   | ❌   |

---

## Intuition in one sentence

> **WL works by repeatedly counting and refining neighborhood structures; any GNN that averages or normalizes loses the ability to count.**

## Final takeaway

* WL is a **gold standard** for graph distinguishability
* GIN reaches **exactly WL power**
* No MPNN can exceed WL
* Failures of WL = fundamental limits of message passing

## 2-WL and Higher-Order GNNs

### Why 1-WL is fundamentally limited

Recall **1-WL**:

* Colors **nodes**
* Looks only at **local neighborhoods**
* Cannot compare **relationships between neighbors**

So it **cannot express pairwise relations** like:

> “Are two neighbors connected to each other?”

This is exactly what breaks 1-WL on many graphs.


### What is 2-WL (a.k.a. Weisfeiler–Lehman on pairs)

#### Key idea:

Instead of coloring **nodes**, **2-WL colors ordered pairs of nodes**.

Each pair $(u, v)$ has a color.


#### Initialization

$$
c^{(0)}(u,v) =
\begin{cases}
\text{edge color} & u = v \text{ or } (u,v)\in E \
\text{non-edge color} & \text{otherwise}
\end{cases}
$$

So we already encode:

* self-pairs
* edges
* non-edges


#### Update rule

For each pair $(u, v)$, collect:
$$
\left{
\left(
c^{(t)}(u, w),
c^{(t)}(w, v)
\right)
;\middle|;
w \in V
\right}
$$

Then hash:
$$
c^{(t+1)}(u,v) =
\text{HASH}\Big(
c^{(t)}(u,v),
;{!{ (c(u,w), c(w,v)) }!}
\Big)
$$

🚨 This captures **paths of length 2** and **connectivity patterns**.


### Concrete example — 2-WL succeeds where 1-WL fails

#### Strongly regular graphs (classic failure of 1-WL)

Properties:

* All nodes same degree
* Same number of common neighbors for adjacent / non-adjacent nodes

👉 1-WL sees **everything identical**

---

#### What 2-WL sees

For a node pair ((u, v)):

* Counts how many nodes connect to both
* Distinguishes:

  * adjacent pairs
  * non-adjacent pairs
  * structurally different neighborhoods

✔ 2-WL can separate many strongly regular graphs
❌ 1-WL cannot


#### How 2-WL maps to GNNs

| WL                    | GNN                |
|-----------------------|--------------------|
| Nodes                 | Node embeddings    |
| Node pairs            | Pair embeddings    |
| Neighborhood multiset | Tensor aggregation |

So a **2-WL-powerful GNN** must:

* Maintain embeddings for **node pairs**
* Aggregate over a **third node**


#### Example: 2-WL-style GNN update

$$
h_{u,v}^{(l+1)} =
\text{MLP}\Big(
h_{u,v}^{(l)},
\sum_{w \in V}
\phi(h_{u,w}^{(l)}, h_{w,v}^{(l)})
\Big)
$$

This is **O(n³)** 😬
That’s the price of expressiveness.


### Higher-order WL (k-WL)

| k | Colors       |
|---|--------------|
| 1 | Nodes        |
| 2 | Node pairs   |
| 3 | Node triples |
| k | k-tuples     |

General rule:
$$
k\text{-WL} ;\text{colors}; V^k
$$

Each increase:

* Exponentially increases power
* Exponentially increases cost


### Higher-order GNN families

#### k-GNN (Morris et al.)

* Explicit k-tuples
* Provably matches k-WL

#### Folklore GNN / Higher-order MPNN

* Tensor-based message passing

#### Subgraph GNNs

* Color induced subgraphs instead of tuples

#### Equivariant GNNs

* Respect permutation symmetry
* Use tensor representations


### Expressiveness hierarchy

$$
\text{GCN / GAT} < \text{GIN} = 1\text{-WL} < 2\text{-WL} < \dots < k\text{-WL}
$$


## Why Regular Graphs Break Everything


### What is a regular graph?

A graph is **k-regular** if:
$$
\deg(v) = k \quad \forall v
$$

Examples:

* Cycles
* Complete graphs
* Many expander graphs


## Why 1-WL fails immediately

### Iteration 0

All nodes same color

### Iteration 1

Each node sees:
$$
{!{ A, A, \dots, A }!} \quad (k \text{ times})
$$

All hashes identical.

### Iteration t

Nothing changes — **perfect symmetry**

🚨 No refinement happens.


## Why GNNs also fail (deeper insight)

A message passing layer computes:
$$
h_v^{(l+1)} = f\Big(
h_v^{(l)},
{!{ h_u^{(l)} : u \in \mathcal{N}(v) }!}
\Big)
$$

In a regular graph:

* All nodes receive the **same multiset**
* All embeddings stay identical

Even with:

* Infinite depth
* Nonlinearity
* Attention


## Why attention does NOT help

GAT computes:
$$
h_v^{(l+1)} =
\sum_{u \in \mathcal{N}(v)}
\alpha_{vu} h_u
$$

But:

* Attention scores are symmetric
* Softmax enforces normalization
* All neighbors are identical

➡️ All weights equal
➡️ Reduces to mean aggregation


## Stronger example: Non-isomorphic regular graphs

Two different 3-regular graphs:

* Same number of nodes
* Same degree
* Different global structure

| Method | Can distinguish? |
|--------|------------------|
| 1-WL   | ❌                |
| GIN    | ❌                |
| GCN    | ❌                |
| GAT    | ❌                |
| 2-WL   | ✔ (often)        |


## Why this is a fundamental limitation

This is **not an implementation issue**.

It is a **theoretical impossibility**:

> Any permutation-invariant, local, message-passing architecture cannot break perfect symmetry.

---

## How researchers break the curse

### Higher-order GNNs (k-WL)

Capture relations between neighbors

### Positional encodings

Inject asymmetry (Laplacian eigenvectors)

### Random features

Break symmetry probabilistically

### Subgraph sampling

Expose global patterns


## One-sentence intuition

> Regular graphs defeat GNNs because **every node lives in the same universe**, and message passing cannot invent asymmetry.

---

## Final summary

### WL side

* 1-WL = node counting
* 2-WL = pairwise structure
* k-WL = higher-order relations

### GNN side

* GIN matches 1-WL
* Higher-order GNNs match k-WL
* Regular graphs expose fundamental symmetry limits


