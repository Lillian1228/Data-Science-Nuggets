### 5.1 ML with Heterogeneous Graphs

- So far we only handle graphs with one edge type
- How to handle graphs with multiple nodes or edge types (a.k.a heterogeneous graphs)?
- Goal: Learning with heterogeneous graphs
  - Relational GCNs
  - Heterogeneous Graph Transformer
  - Design space for heterogeneous GNNs


#### 5.1.1 Heterogeneous Graphs

- Motivation: A graph could have multiple types of nodes and edges.

    Example: 2 types of nodes (paper, author) + 2 types of edges (cite, like) &rarr; 8 possible **relation types (node start, edge, node_end)** !

    <img src="src/L5/5.1.1.png" width="400"> 

    - We use **relation type to describe an edge** (as opposed to edge type)
    - Relation type better captures the interaction between nodes and edges

- Definition: A heterogeneous graph is defined as $G=(V,R,T,E)$
  - Nodes $v_i \in V$ with node types $T(v_i)$
  - Edge types $e \in E$
  - Relations with relation types $(v_i,e,v_j) \in E$

- Examples
  1. Biomedical Knowledge Graphs: (fulvestrant, Treats, Breast Neoplasms)

    <img src="src/L5/5.1.2.png" width="500"> 

  2. Event Graphs: (UA689, Origin, LAX)

    <img src="src/L5/5.1.3.png" width="400">  

  3. E-Commerce Graph
       - Node types: User, Item, Query, Location, ...
       - Edge types: Purchase, Visit, Guide, Search, ...
       - Different node type's features spaces can be different!

    <img src="src/L5/5.1.4.png" width="400">   

  4. Academic Graph
       - Node types: Author, Paper, Venue, Field, ...
       - Edge types: Publish, Cite, ...
       - Benchmark dataset: Microsoft Academic Graph
  
    <img src="src/L5/5.1.5.png" width="400">    

- Observation: We can also treat types of nodes and edges as features

    Example: Add a one-hot indicator for nodes and edges
  - Append feature [1, 0] to each “author node”; Append feature [0, 1] to each “paper node”
  - Similarly, we can assign edge features to edges with different types
  - Then, a heterogeneous graph reduces to a standard graph

- When do we need a heterogeneous graph?

    Case 1: Different node/edge types have different shapes of features
    - An “author node” has 4-dim feature, a “paper node” has
5-dim feature
    
    Case 2: We know different relation types
represent different types of interactions
    - (English, translate, French) and (English, translate, Chinese) require different models

    Ultimately, heterogeneous graph is a more expressive graph representation
    - Captures different types of interactions between entities

    But it also comes with costs:
    - More expensive (computation, storage)
    - More complex implementation

    There are many ways to convert a heterogeneous graph to a standard graph (that is, a homogeneous graph).

#### 5.1.2 Relational GCN (RGCN)

<img src="src/L5/5.1.6.png" width="400"> 

We will extend GCN to handle heterogeneous graphs with multiple edge/relation types

We start with a directed graph with one relation. How do we run GCN and update the representation of the target node A on this graph?

<img src="src/L5/5.1.7.png" width="500"> 

- What if the graph has multiple relation types?

    Use different neural network weights for different relation types.

    <img src="src/L5/5.1.8.png" width="400"> 

    <img src="src/L5/5.1.9.png" width="300">  

    Introduce a set of neural networks for each relation type!

    <img src="src/L5/5.1.10.png" width="300">

- Relational GCN: Definition

    <img src="src/L5/5.1.11.png" width="400">

    How to write this as Message + Aggregation?
    - Message: 
      - Each neighbor of a given relataion: $m^{(l)}_{(u,r)}=1/c_{(u,r)}W^{(l)}_rh^{(l)}_u$, normalized by node degree of the relation $c_{(u,r)}=|N^r_v|$.
      - Self-loop: $m^{(l)}_v=W^{(l)}_0h^{(l)}_v$
    - Aggregation: Sum over messages from neighbors and self-loop, then apply activation
    
      <img src="src/L5/5.1.12.png" width="400">

- Scalability
  - Each relation has L matrices: $W^{(1)}_r,W^{(2)}_r,...W^{(L)}_r$
  - The size of each $W^{(l)}_r$ is $𝑑^{(𝑙+1)} × 𝑑^{(𝑙)}$, where $𝑑^{(𝑙)}$ is the hidden dimension in layer l.

    &rarr; Rapid growth of the number of parameters w.r.t number of relations! Overfitting becomes an issue!
  - Two methods to regularize the weights $W^{(l)}_r$
    1. Use block diagonal matrices
    2. Basis/Dictionary learning
   
- Block Diagonal Matrices

    key insight: make the weights sparse!

    Use block diagnonal matrices for $W_r$

    <img src="src/L5/5.1.13.png" width="400">

    If use 𝐵 low-dimensional matrices, then # param reduces from $𝑑^{(𝑙+1)} × 𝑑^{(𝑙)}$ to $B × 𝑑^{(𝑙+1)}/B × 𝑑^{(𝑙)}/B$

- Basis Learning

    Key insight: Share weights across different relations!

    Represent the matrix of each relation as a **linear combination of basis transformations**

    $W_r = \sum_{b=1}^Ba_{rb} \cdot V_b$, where $V_b$ is shared across all relations
    - $V_b$ are the basis matrices
    - $a_{rb}$ is the importance weight of matrix $V_b$

    Now each relation only needs to learn $[a_{rb}]_{b=1}^B$ which is B scalars.

- Examples

  1. Entity/Node Classification

    Goal: Predict the label of a given node

    RGCN uses the representation of the final layer:
    - If we predict the class of node A from k classes
    - Take the final layer (prediction head): $h^{(L)}_A \in \mathbb{R}^k$, each item in $h^{(L)}_A$ represents the probability of that class

  2. Link Prediction

    Link prediction split: graph is split with 4 categories of edges (training message / training supervision / validation / test)

    Every edge also has a relation type, this is independent of the 4 categories.
    
    In a heterogeneous graph, the homogeneous graphs formed by every single relation also have the 4 splits.

    <img src="src/L5/5.1.14.png" width="500">  

- RGCN for Link Prediction

    Assume (𝑬, 𝒓𝟑 , 𝑨) is training supervision edge, all the other edges are training message edges.
    
    Use RGCN to score (𝑬, 𝒓𝟑 , 𝑨):

    <img src="src/L5/5.1.15.png" width="200"> 

    - Take the final layer of E and A: $h^{(L)}_E$ and $h^{(L)}_A \in \mathbb{R}^d$ 
    - Relation-specific score function $f_r: \mathbb{R}^d \times \mathbb{R}^d$ &rarr; $\mathbb{R}$
      - one example $f_{r_1}(h_E,h_A)=h^T_EW_{r_1}h_A$, $W_{r_1} \in \mathbb{R}^{d \times d}$
  
    Training:

    <img src="src/L5/5.1.16.png" width="500"> 
    
    1. Use GNN model to score negative edge
    2. Optimize a standard cross entropy loss: 
       - Maximize the score of training supervision edge
       - Minimize the score of negative edge

        <img src="src/L5/5.1.17.png" width="400">  
    
    Evaluation: Use training message edges & training supervision edges to predict validation edges. Validation time as an example, same at the test time.

    <img src="src/L5/5.1.18.png" width="500"> 

    <img src="src/L5/5.1.19.png" width="500"> 

- Benchmark for Heterogeneous Graphs

    [ogbn-mag](https://ogb.stanford.edu/docs/nodeprop/) from Microsoft Academic Graph (MAG)
    - 4 types of entities: Papers (736k nodes), Authors (1.1.m nodes), Institutions (9k nodes), Fields of study (60k nodes)

    <img src="src/L5/5.1.20.png" width="500"> 

    - 4 directed relations:
      - An author is affiliated with an institution
      - An author writes a paper
      - A paper cites a paper
      - A paper has a topic of a field of study

    - Prediction task
      - Each paper has a 128-dimensional word2vec feature vector
      - Given the content, references, authors, and author affiliations
from ogbn-mag, predict the venue of each paper
      - 349-class classification problem due to 349 venues considered
    - Time-based dataset splitting
      - Training set: papers published before 2018
      - Test set: papers published after 2018
    - Benchmark results

        <img src="src/L5/5.1.21.png" width="500"> 

        SOTA method: SeHGNN
        - ComplEx (Next lecture) + Simplified GCN (Lecture 17)

- Summary of RGCN

    - Relational GCN, a graph neural network for heterogeneous graphs
    - Can perform entity classification as well as link prediction tasks.
    - Ideas can easily be extended into RGNN (RGraphSAGE, RGAT, etc.)
    - Benchmark: ogbn-mag from Microsoft Academic Graph, to predict paper venues


#### 5.1.3 Heterogeneous Graph Transformer



#### 5.1.4 Design Space for Heterogenous GNNs






### 5.2 Knowledge Graph Embeddings

#### 5.2.1 Motivation


#### 5.2.2 Knowledge Graph Completion: TransE and TransR




#### 5.2.3 Knowledge Graph Completion: DistMult




#### 5.2.4 Knowledge Graph Completion: ComplEx


#### 5.2.5 Knowledge Graph Embeddings in Practice



### 5.3 Reasoning over Knowledge Graphs


#### 5.3.1 Reasoning in KGs using Embeddings

#### 5.3.2 Answering PRedictive Queries on KGs

#### 5.3.3 Query2Box: Reasoning over KGs using Box Embeddings