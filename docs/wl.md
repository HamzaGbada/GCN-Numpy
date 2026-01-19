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

