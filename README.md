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

* Zhang, Si & Tong, Hanghang & Xu, Jiejun & Maciejewski, Ross. (2019). Graph convolutional networks: a comprehensive review. Computational Social Networks. 6. 10.1186/s40649-019-0069-y. 


