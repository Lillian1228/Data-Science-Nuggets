### 2.1 Graph Neural Netowrks (GNNs)

- Node Embeddings Recap: Map nodes to 𝑑-dimensional embeddings such that similar nodes in the graph are embedded close together.

    <img src="src/1.2_5.png" width="400">
    <img src="src/1.2_6.png" width="400">

- Recap: Shallow Encoders

    Simplest encoding approach: encoder is just an embedding-lookup. ENC(v) = $z_v$ = $Z \cdot v$

    <img src="src/1.2_7.png" width="400">
    <img src="src/1.2_8.png" width="400">

    Limitations of shallow embedding methods:
    <img src="src/2.1.1.png" width="400"> 

- Today: Deep Graph Encoders   

    **ENC(v) = multiple layers of non-linear transformations based on graph structure.**

    Note: all the deep encoders can be combined with node similarity functions defined in the Lecture 3.
    <img src="src/2.1.2.png" width="400"> 

    Tasks on Networks:
    - Node classification: predict the type of a given node
    - Link prediction: predict whether two nodes are linked
    - Community detection: identify densely linked clusters of nodes
    - Network similarity: how similar are two (sub)networks

#### 2.1.1 Basics of Deep Learning

Supervised learning: we are given input 𝒙, and the goal is to predict label 𝒚. 

Input x can be: vectors of real numbers, sequences (natural language), matrices (images), graphs (potentially with node and edge features).

<img src="src/2.1.3.png" width="500"> 

- Multi-layer Perceptron (MLP): each layer of MLP conbines linear transformation and non-linearity.
  
    <img src="src/2.1.4.png" width="500"> 

    Suppose x is 2-dimensional, with entries x1 and x2:
    
    <img src="src/2.1.5.png" width="400"> 

- Summary

    <img src="src/2.1.6.png" width="400"> 

#### 2.1.2 Deep Learning for Graphs

Problem Setup:

<img src="src/2.1.7.png" width="400">

- A Naive Approach: Join adjacency matrix and features. Feed them intoa deep neural net.

    <img src="src/2.1.8.png" width="600">   

    Issues with this idea:
  - O(|V|) parameters
  - Not applicable to graphs of different sizes
  - Sensitive to node ordering

- Idea: Convolutional Netowrks

    <img src="src/2.1.9.png" width="600">     

    Goal is to generalize convolutions beyonod simple lattices leveraging node features/attributes (i.e. text, images).

    But for real-world graphs, there is no fixed notion of locality or sliding window on the graph. Graph is permutation invariant.

- Permutation Invariance

    Graph does not have a canonical order of the nodes! We can have many different order plans.

    <img src="src/2.1.10.png" width="600">   

    **Graph and node representations should be the same for order plan 1 and 2.**

    <img src="src/2.1.11.png" width="500">  

    <img src="src/2.1.12.png" width="500">  

- Permutation Equivariance

    For node representation, consider we learn a function $f$ that maps a graph $G=(A,X)$ to a matrix of shape (m,d).

    If the output vector of a node at the same position in the graph remains unchanged for any order plan, we say $f$ is **permutation equivariant**.

    <img src="src/2.1.13.png" width="500"> 

    <img src="src/2.1.14.png" width="500"> 

- GNN Overview

    GNNs consist of multiple permutation equivariant/invariant functions.

    <img src="src/2.1.15.png" width="500"> 

    Other neural network architectures i.e. MLPs are not permutation invariant/equivariant. Changing the order of the input leads to different outputs. This explains why the naive MLP approach fails for graphs.

    <img src="src/2.1.16.png" width="500"> 

    Therefore, we need to design GNNs that are permutation invariant/equivariant by passing and aggregating information from neighbors.

#### 2.1.3 Graph Convolution Networks

General Idea: Node's neighborhood defines a computation graph. Learn how to propagate information across the graph to comupte node features.

<img src="src/2.1.17.png" width="500"> 

- Aggregate Neighbors

    Key idea: Generate node embeddings based on local network neighborhoods. Information is aggregated from their neighbors using neural networks.

    <img src="src/2.1.18.png" width="500"> 

    <img src="src/2.1.19.png" width="500">    

- Deep Model: Many Layers

    Model can be of arbitrary depth:
    - Nodes have embeddings at each layer
    - Layer 0 embedding of node $v$ is its input features $x_v$
    - Layer-k embedding gets info from nodes that are k hops away
  
    <img src="src/2.1.20.png" width="500">  

- Neighborhood Aggreagation: Key distinctions are in how different approaches aggregate info across the layers.

    Basic approach: average info from neighbors and apply a neural network.

    <img src="src/2.1.21.png" width="500">  

    The invariance and equivariance properties for a GCN:
    - Given a node, the GCN that computes its embedding is permutation invariant.

    <img src="src/2.1.22.png" width="500">  

    - Considering all nodes in a graph, GCN computation is permutation equivariant.

    <img src="src/2.1.23.png" width="500"> 

- Training the Model: Model Parameters

    $h^k_v$: the hidden representation of node $v$ at layer $k$
    - $W_k$: weight matrix for neighborhood aggregation
    - $B_k$: weight matrix for transforming hidden vector of self

    <img src="src/2.1.25.png" width="500"> 

    We can feed these embeddings into any loss function and run SGD to train the weight parameters.

- Training the Model: Matrix Formulation
  
  Many aggregations can be performed efficiently by (sparse) matrix operations.

    <img src="src/2.1.26.png" width="600"> 
    <img src="src/2.1.27.png" width="500"> 

    Note: not all GNNs can be expressed in matrix form, when aggregation function is complex.
- Training the Model: Loss function

    <img src="src/2.1.24.png" width="500"> 
    
  - Unsupervised Training: "Similar" nodes have similar embeddings

  <img src="src/2.1.28.png" width="500">  

  - Supervised Training: Directly train the model for a supervised task 
  
    i.e. node classification on safe or toxic drug using cross entropy loss

  <img src="src/2.1.29.png" width="500">

- Model Design Overview

    1. Define a neighborhood aggregation function
    2. Define a loss function on the embeddings
    3. Train on a set of nodes, i.e. a batch of compute graphs
    4. Generate embeddings for nodes as needed, even for nodes we never trained on.

- Inductive Capability
    - The same aggregation parameters are shared for all nodes. 
    - The number of model parameters is sublinear in |V| and we can generalize to unseen nodes.
  
  <img src="src/2.1.30.png" width="500">  

    - New Nodes: Many application settings constantly encounter previously unseen nodes, i.e. Reddit, YouTube, Google Scholar. So we need to generate new embeddings "on the fly".
    
    - New Graphs: Inductive node embedding also allows generalizing to entirely unseen graphs. i.e. train on protein interaction graph from model organism A and generate embeddings on newly collected data about organism B.

#### 2.1.3 GNNs subsume CNNs

How do GNNs compare to prominent architectures such as Convolutional Neural Nets?

  <img src="src/2.1.31.png" width="500"> 

  <img src="src/2.1.32.png" width="500">

Key difference is that, we can learn different $W^u_l$ for different "neighbor" $u$ for pixel $v$ on the image. The reason is we can pick an order for the 9 neighbors using **relative position** to the center pixel: {(-1,-1). (-1,0), (-1, 1), …, (1, 1)}

CNN can be seen as a special GNN with fixed neighbor size and ordering:
- The size of the filter is pre-defined for a CNN.
- The advantage of GNN is it processes arbitrary graphs with different degrees for each node.
- CNN is not permutation invariant/equivariant. Switching the order of pixels will leads to different outputs.

<img src="src/2.1.33.png" width="500">

### 2.2 GNN Design Space

#### 2.2.1 A General Perspective on GNNs

#### 2.2.2 Designing a Single Layer of a GNN

#### 2.2.3 Stacking Layers of a GNN





### 2.3 GNN Training Pipeline

#### 2.3.1 Graph Augmentation for GNNs

#### 2.3.2 Training GNNs


#### 2.3.3 Setting up GNN Prediction Tasks