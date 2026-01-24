## 🧠 Role

You are a senior ML researcher and software architect with deep knowledge of Graph Neural Networks, MPNN theory, and software design patterns.

You are writing documentation for **educational clarity and theoretical correctness**, not marketing.

---

## 🎯 Goal

Update **`README.md` only** to fully document this repository by:

1. Explaining the **Message Passing Neural Network (MPNN)** framework
2. Explaining the **Template Method design pattern**
3. Showing how **each GNN (GCN, GAT, GIN, GraphSAGE)** is an **MPNN specialization**
4. Explicitly mapping:

   * **Mathematical equations**
   * → **MPNN components (message / aggregate / update)**
   * → **Concrete implementation in this codebase**
5. Pointing readers to existing `docs/*.md` files for **model-specific mathematical detail**

🚫 **Do NOT modify anything under `docs/`**
🚫 **Do NOT change code or project structure**

---

## 📁 Context (Important)

The project already contains detailed mathematical documents:

```
docs/
├── gcn.md
├── gat.md
├── gin.md
├── GraphSAGE.md
├── mpnn.md
├── wl.md
```

These files:

* Contain the **exact math used in the NumPy implementations**
* Must be referenced from the README
* Must NOT be duplicated or edited

The README should act as a **conceptual and architectural map**, not a replacement.

---

## ✨ Required README Additions & Structure

### 1️⃣ High-level Philosophy Section

Add a section explaining:

* Why all GNNs are unified under **MPNN**
* Why the project prioritizes:

  * NumPy
  * explicit matrix math
  * manual backprop
* Why this repo is meant for **learning and theory–code alignment**

---

### 2️⃣ MPNN as Mathematics → Software

Add a section titled something like:

> **From MPNN Theory to Software Architecture**

This section must:

* Present the **general MPNN equations**
* Explain that the project implements MPNN using the **Template Method pattern**
* Emphasize that:

  * `forward()` is fixed
  * specialization happens via `message`, `aggregate`, `update`
  * `backward()` is manually implemented per layer

No code listings are required — focus on **conceptual clarity**.

---

### 3️⃣ Template Method Pattern Explanation (Explicit)

Add a dedicated section explaining:

* What the **Template Method pattern** is
* Why it is the *correct* design pattern for MPNN
* How it enforces:

  * a fixed execution pipeline
  * mathematical correctness
  * controlled extensibility

Clarify the one-to-one mapping:

| MPNN Concept | Design Pattern Role |
| ------------ | ------------------- |
| Message      | Hook method         |
| Aggregate    | Hook method         |
| Update       | Hook method         |
| Forward      | Template method     |

---

### 4️⃣ GNN → MPNN Mapping Table (Central Requirement)

Add a **single, clear table** mapping each implemented GNN to the MPNN abstraction:

| Model     | Message              | Aggregate       | Update              | MPNN Specialization   | Reference           |
| --------- | -------------------- | --------------- | ------------------- | --------------------- | ------------------- |
| GCN       | Fixed linear message | Normalized sum  | Linear + activation | Aggregation-centric   | `docs/gcn.md`       |
| GAT       | Attention-weighted   | Weighted sum    | Linear + activation | Message + aggregation | `docs/gat.md`       |
| GIN       | Identity             | Sum (injective) | MLP                 | Update-centric        | `docs/gin.md`       |
| GraphSAGE | Identity             | Mean / Max      | Linear + activation | Aggregation-centric   | `docs/GraphSAGE.md` |

Each row must:

* Describe **what changes mathematically**
* State **where the specialization occurs conceptually**
* Link to the corresponding `docs/*.md`

---

### 5️⃣ Design–Math Alignment Section

Add a section clarifying that:

* The implementation is **not a framework**
* It mirrors the **mathematical definition directly**
* Each GNN differs only in:

  * how messages are computed
  * how neighbors are aggregated
  * how node states are updated

Emphasize that this mirrors **Gilmer et al. (2017)** exactly.

---

### 6️⃣ Documentation Index (Do Not Duplicate Content)

Add or refine a table that clearly states:

| Document            | Purpose                      |
| ------------------- | ---------------------------- |
| `docs/mpnn.md`      | General MPNN theory          |
| `docs/gcn.md`       | Exact GCN math               |
| `docs/gat.md`       | Exact GAT math               |
| `docs/gin.md`       | Exact GIN math               |
| `docs/GraphSAGE.md` | Exact GraphSAGE math         |
| `docs/wl.md`        | WL test & GIN expressiveness |

Make it clear:

> “The README explains *how everything fits together*; the docs explain *the math in detail*.”

---

### 7️⃣ References Section (Consolidated)

Ensure the README ends with a **clean reference list**, citing:

* Gilmer et al. — MPNN
* Kipf & Welling — GCN
* Veličković et al. — GAT
* Xu et al. — GIN
* Hamilton et al. — GraphSAGE
* Weisfeiler–Lehman test

---

## 🧼 Constraints

* Modify **only `README.md`**
* Do NOT touch `docs/*`
* Do NOT add code
* Do NOT change math
* Keep the USage part in the current `README.md`
* Keep tone **academic, clear, and instructional**
* Assume the reader knows basic linear algebra and ML

---

## 📤 Output Requirements

* Show the **full updated `README.md`**
* Preserve existing content where possible
* Integrate new sections cleanly
* Do not include explanations outside the README text

---

**Proceed to update `README.md` now.
Focus on clarity, correctness, and tight alignment between mathematics and design.**

