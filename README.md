## Mathematical Details of Graph Convolutional Network (GCN) for Node Classification

### Step 1: Data Representation
- **Adjacency Matrix \( A \)**:
  - Represents the graph structure where \( A_{ij} = 1 \) if there is an edge between nodes \( i \) and \( j \), and \( A_{ij} = 0 \) otherwise.
- **Input Feature Matrix \( X \)**:
  - Represents node features where each row corresponds to a node and each column corresponds to a feature.

### Step 2: Initialization
- **Weight Matrices \( W^{(l)} \)**:
  - Initialize weight matrices for each layer \( l \) of the GCN.
- **Bias Vectors \( b^{(l)} \)** (optional):
  - Optionally, initialize bias vectors for each layer.

### Step 3: Forward Propagation
- **Normalized Graph Laplacian \( \tilde{L} \)**:
  - Compute the normalized graph Laplacian: \( \tilde{L} = I - D^{-\frac{1}{2}} A D^{-\frac{1}{2}} \).
- **Graph Convolution Operation**:
  - Compute the node representation matrix at layer \( l+1 \): 
    \[ H^{(l+1)} = \sigma(\tilde{L} H^{(l)} W^{(l)}) \]
  - \( \sigma \) is the activation function.

### Step 4: Loss Calculation
- **Loss Function**:
  - Compute the loss function based on the task, such as node classification.
  - Common choice: Categorical cross-entropy loss.

### Step 5: Backpropagation
- **Gradient Computation**:
  - Compute the gradients of the loss function with respect to the model parameters using backpropagation.
  - Example: \( \frac{\partial L}{\partial W} = \frac{1}{N} (X^T A^T) (\hat{Y} - Y) \).
- **Parameter Update**:
  - Update the model parameters using gradient descent or another optimization algorithm.

### Step 6: Training Loop
- **Iteration**:
  - Repeat steps 3-5 iteratively until convergence or for a fixed number of epochs.

### Mathematical Equations

### Graph Convolution Operation
\[ H^{(l+1)} = \sigma(\tilde{L} H^{(l)} W^{(l)}) \]

### Normalized Graph Laplacian
\[ \tilde{L} = I - D^{-\frac{1}{2}} A D^{-\frac{1}{2}} \]

### Loss Function (Categorical Cross-Entropy)
\[ L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} Y_{ij} \log(\hat{Y}_{ij}) \]

### Implementation Details
- Utilize NumPy to implement the GCN architecture and operations.
- Implement helper functions for tasks such as normalization and activation functions.
- Monitor the training process by tracking the loss curve over epochs.

By understanding these mathematical details and following the outlined steps, you can implement a basic version of GCN from scratch for node classification tasks. Adjustments and optimizations can be made based on specific requirements and computational resources.



<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">GCN-Numpy</h1>

</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#usage">Usage</a></li>
        <li><a href="#note">Note</a></li>
      </ul>
    </li>
    <li><a href="#references">References</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This is concise implementation of Graph Convolution Network (GCN) for educational purpose using **Numpy** and **Networkx**.

### Built With

* [Python](https://www.python.org/)
* [Numpy](https://numpy.org/)
* [NetworkX](https://networkx.org/)






<!-- GETTING STARTED -->
## Getting Started


### Usage

1. Clone the repo
   ```sh
   $ git clone https://github.com/HamzaGbada/GCN-Numpy.git
   ```
2. Install the requirement libraries
   ```sh
   $ pip install -r requirements.txt
   ```
3. Testing
    You can build the model without training
    ```shell script
    $ python main.py
    ```
    Or you can train it directly
    ```shell script
    $ python train.py
    ```
    

### Note
Here I used Networkx [Binomial graph](https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.binomial_graph.html) (Erdős-Rényi graph) for generating random graphs as dataset it params are choosing randomly.
You are free to change it, I already test it on [Zachary’s Karate Club graph](http://vlado.fmf.uni-lj.si/pub/networks/data/ucinet/ucidata.htm#zachary) and The [Turan Graph](https://mathworld.wolfram.com/TuranGraph.html).

<!-- references -->
## References

* Kipf, Thomas & Welling, Max. (2016). Semi-Supervised Classification with Graph Convolutional Networks.  


