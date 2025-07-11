### 6.1 Frequent Subgraph Mining with GNNs

Building Blocks of Networks - Subgraphs:

They have the power to **characterize and discriminate** networks.

<img src="src/L6/6.1.1.png" width="500">  

Plan for Today:
1. Subgraphs and motifs
    - Definition
    - Determining Motif Significance
  2. Neural Subgraph Representations
  3. Mining Frequent Motifs

#### 6.1.1 Subgraphs and motifs

Given graph $G=(V, E)$, two ways to formalize "network building blocks":
- Def 1: **Node-induced subgraph** (or "induced subgraph"): Take subset of the nodes and all edges induced by the nodes.
  
  $G^′ = (V^′, E^′)$ is a node induced subgraph iff
  - $V' \subset V$
  - $E' = \{(u,v) \isin E | u, v \isin V'\}$
  - $G'$ is the subgraph of G **induced by $V'$**

- Def 2. **Edge-induced subgraph** (or "non-induced subgraph" or just "subgraph"): Take subset of the edges and all corresponding nodes.

  $G^′ = (V^′, E^′)$ is an edge induced subgraph iff
  - $E' \subset E$
  - $V' = \{v \isin V | (u, v) \isin E'$ for some $u \}$

The best definition depends on the domain! For example:
- Chemistry: node-induced (functional groups)
- Knowledge graphs: Often edge-induced (focus is on edges representing logical relations)

The preceding definitions define subgraphs when V′ ⊆ V and E′ ⊆ E, i.e. nodes and edges are taken from the original graph G. 

- What if V′ and E′ come from a totally different graph? e.g.

    <img src="src/L6/6.1.2.png" width="300"> 

    We would like to say that $G_1$ is “contained in” $G_2$

- Graph Isomorphism

    - Problem: Check whether two graphs are identical.

    - $G_1 = (V_1, E_1)$ and $G_2 = (V_2, E_2)$ are **isomorphic** if there exists a bijection $f: V_1 → V_2$ such that $(u, v) ∈ E_1$ iff $(f(a), f(b)) ∈ E_2$
    - $f$ is called the isomorphisim:

        <img src="src/L6/6.1.3.png" width="400">  

    - We do not know if graph isomorphism is NP-hard, nor is any polynomial algorithm found for solving graph isomorphism.

- Subgraph Isomorphism
    - $G_2$ is **subgraph-isomorphic** to $G_1$ if some subgraph of $G_2$ is isomorphic to $G_1$
    - We also commonly say $G_1$ is a subgraph of $G_2$
    - We can use either the node-induced or edge-induced
    - This problem is NP-hard

    <img src="src/L6/6.1.4.png" width="400">

- Case Example of Subgraphs

   All non-isomorphic, connected, undirected graphs of size 4:

   <img src="src/L6/6.1.5.png" width="300">

   All non-isomorphic, connected, directed graphs of size 3:

    <img src="src/L6/6.1.6.png" width="400"> 

- Network Motifs: “recurring, significant patterns of interconnections”

    How to define a network motif:
    - Pattern: Small (node-induced) subgrpah
    - Recurring: Found many times, i.e., with high frequency
    - Significant: More frequent than expected, i.e., in randomly generated graphs?

    <img src="src/L6/6.1.10.png" width="400">  

    Why do we need Motifs?
    - Help us understand how graphs work
    - Help us make predictions based on presence or lack of presence in a graph dataset
  
    Examples:
    - Feed-forward loops: found in networks of neurons, where they neutralize “biological noise”
    - Parallel loops: found in food webs
    - Single-input modules: found in gene control networks

    <img src="src/L6/6.1.7.png" width="100">   
    <img src="src/L6/6.1.8.png" width="100"> 
    <img src="src/L6/6.1.9.png" width="200">

- Subgraph Frequency

    Let $G_Q$ be a small graph and $G_T$ be a target graph dataset.

  1. Graph-level Subgraph Frequency Definition: Frequency of $G_Q$ in $G_T$

       - number of unique subsets of nodes $V_T$ of $G_T$ for which the subgraph of $G_T$ induced by the nodes $V_T$ is isomorphic to $G_Q$ 

        <img src="src/L6/6.1.11.png" width="400"> 

    Let $G_Q$ be a small graph, $v$ be a node in $G_Q$ (the "anchor") and $G_T$ be a target graph dataset.

  2. Node-level Subgraph Frequency Definition:

        - The number of nodes $u$ in $G_T$ for which some subgraph of $G_T$ is isomorphic to $G_Q$ and the isomorphism maps $u$ to $v$

        - Let $(G_Q, v)$ be called a **node-anchored** subgraph
        - Robust to outliers
  
        <img src="src/L6/6.1.12.png" width="300">  
    
    What if the dataset contains **multiple graphs**, and we want to compute frequency of subgraphs in the dataset?

    Idea: Treat the dataset as a giant graph $G_T$ with disconnected components corresponding to individual graphs.

- Defining Motif Significance

    To define significance we need to have a null model (i.e., point of comparison)

    Key idea: Subgraphs that **occur in a real network much more often** than in a **random** network have functional significance

    <img src="src/L6/6.1.13.png" width="400"> 

    - Defining Random Graphs
  
        Erdős–Rényi (ER) random graphs $G_{n,p}$: undirected graph on $n$ nodes where each edge $(u, v)$ appears i.i.d. with probability $p$

        Can be disconnected: 

        <img src="src/L6/6.1.14.png" width="400"> 

    - New Model: Configuration Model
      - Goal:  Generate a random graph with a given **degree sequence $k_1, k_2, … k_N$**
      - Useful as a “null” model of networks: We can compare the real network $G^{real}$ and a “random” $G^{rand}$ which has the same degree sequence as $G^{real}$
  
      <img src="src/L6/6.1.15.png" width="500">  

    - Alternative for Spokes: Switching
      - Start from a given graph G
      - Repeat **the switching step** $Q \cdot|E|$ times:
        - Select a pair of edges A &rarr; B, C &rarr; D at random
        - **Exchange** the endpoints to give A &rarr; D, C &rarr; B
        - Exchange edges only if no multiple edges or self-edges are generated
      - Result: A randomly rewired graph with the same node degrees and randomly rewired edges
      - Q is chosen large enough (e.g. Q = 100) for the process to converge.

- Evaluating Motif Significance
  - Z-score $Z_i$ captures statistical significance of motif i:

    $Z_i = (N^{real}_i-\bar{N^{rand}_i})/std(N^{rand}_i)$
    - $N^{real}_i$ is # of motif i in graph $G^{real}$
    - $N^{rand}_i$ is average # of motif i in random graph instances
  
    For each subgraph, z-score metric is capable of classifying the subgraph “significance”:
    - Negative values indicate **under-representation**
    - Positive values indicate **over-representation**

  - **Network significance profile (SP)**: $SP_i = Z_i/\sqrt{\sum_{j}Z^2_j}$
    - SP is a vector of normalized Z-scores, with values for all subgraph types.
    - The dimension depends on number of motifs considered
    - SP emphasizes relative significance of subgraphs:
      - Important for comparison of networks of different sizes
      - Generally, larger graphs display higher Z-scores

  - Compare profiles of different graphs with random graphs:
    - Regulatory network (gene regulation)
    - Neuronal network (synaptic connections)
    - World Wide Web (hyperlinks between pages)
    - Social network (friendships)
    - Language networks (word adjacency)

    <img src="src/L6/6.1.16.png" width="500">

  - Summary: Detecting Motifs

      Intuition: Motifs are overrepresented in a network when compared to random graphs.
    - Step 1: **Count subgraphs i** in the given graph ($G^{real}$)
    - Step 2: **Generate random graphs** with similar statistics (e.g. number of nodes, edges, degree sequence), and count subgraphs in the random graph
      - Null model: Each $G^{rand}$ has the same #(nodes), #(edges) and degree distribution as $G^{real}$.
    - Step 3: **Use statistical measures** like Z-score to evaluate
how significant is each motif
        -  $Z_i = (N^{real}_i-\bar{N^{rand}_i})/std(N^{rand}_i)$
        -  **High Z-score**: Subgraph i is a **network motif of G**

- Summary
    - Subgraphs and motifs are the building blocks of graphs
    - Subgraph isomorphism and counting are NP-hard
    - Understanding which motifs are frequent or significant in a dataset gives insight into the unique characteristics of that domain
    - Use random graphs as null model to evaluate the significance of motif via Z-score

#### 6.1.2 Neural Subgraph Matching

- Subgraph Matching

    Given: large target graph (can be disconnected) and query graph (connected)

    Decode: Is query graph a subgraph in the target graph?

    <img src="src/L6/6.1.17.png" width="300"> 

- Isomorphism as an ML Task

    - Intuition: Exploit the **geometric shape** of embedding space to capture the properties of subgraph isomorphism

    - Idea: Use GNN to predict subgraph isomorphism

    - Task Setup: Consider a **binary** prediction: Return True if query is isomorphic to a subgraph of the target graph, else return False

    <img src="src/L6/6.1.18.png" width="500"> 

- Neural Architecture for Subgraphs

  1. We are going to work with **node-anchored definitions**
  2. We are going to work with **node-anchored neighborhoods**
  3. Use GNN to obtain representations of u and v
  4. Predict if node u’s neighborhood is isomorphic to node v’s neighborhood

    <img src="src/L6/6.1.19.png" width="500"> 

    - Why Anchor?

        Recall node-level frequency definition: The number of nodes $u$ in $G_T$ for which some subgraph of $G_T$ is isomorphic to $G_Q$ and the isomorphism maps $u$ to $v$

        We can compute embeddings for $u$ and $v$ using GNN, then use embeddings to decide if neighborhood of $u$ is isomorphic to subgraph of neighborhood of $v$.

        We not only predict if there exists a mapping, but also identify corresponding nodes ($u$ and $v$)!

- Decomposing $G_T$ into Neighborhoods

   For each node in $G_T$:
   - Obtain a **k-hop neighborhood** around the anchor
   - Can be performed using **breadth-first search** (BFS)
   - The depth k is a hyper-parameter (e.g. 3). Larger depth results in more expensive model

   Same procedure applies to $G_Q$, to obtain the neighborhoods.

   We embed the neighborhoods using a GNN, by computing the **embeddings for the anchor** nodes in their respective neighborhoods.

- Order Embedding Space

    - Intuition: subgraph is to the lower-left of its supergraph (in 2D)
    - Idea: Map graph A to a point $z_A$ into a highdimensional (e.g. 64-dim) embedding space, such that $z_A$ is **non-negative in all dimensions**

    <img src="src/L6/6.1.20.png" width="500">  

    - Why Order Embedding Space?

      **Subgraph isomorphism relationship** can be nicely encoded in order embedding space.
      - Transitivity: if $G_1$ is a subgraph of $G_2$, $G_2$ is a
subgraph of $G_3$, then $G_1$ is a subgraph of $G_3$
      - Anti-symmetry: if $G_1$ is a subgraph of $G_2$, and $G_2$ is
a subgraph of $G_1$, then $G_1$ is isomorphic to $G_2$
      - Closure under intersection: the trivial graph of 1 node is a subgraph of any graph
      - All properties have their counter-parts in the order embedding space

      <img src="src/L6/6.1.21.png" width="500"> 

- Order Constraint

    We use a GNN to learn to embed neighborhoods and preserve the order embedding structure.

    What loss function should we use, so that the learned order embedding reflects the subgraph relationship?

    &rarr; We design loss functions based on the **order constraint**:

    - Order constraint specifies the ideal order embedding
property that reflects subgraph relationships
    - We specify the order constraint to **ensure that the subgraph properties are preserved** in the order embedding space

        <img src="src/L6/6.1.22.png" width="500"> 

    - Loss Function: GNN Embeddings are learned by minimizing a **max-margin loss**
      - Let $E(G_q, G_t)=\sum_{i=1}^D(max(0, z_q[i]-z_t[i]))^2$ be the "margin" between $G_q$ and $G_t$
      - To learn the correct order embeddings, we want to learn $z_q$, $z_t$ such that
        - $E(G_q, G_t)=0$ when $G_q$ **is** a subgraph of $G_t$
        - $E(G_q, G_t)>0$ when $G_q$ **is not** a subgraph of $G_t$
        
        <img src="src/L6/6.1.23.png" width="500"> 

- Training Neural Subgraph Matching

    - Overview
      - Construct training examples ($G_q$, $G_t$) where half the time, $G_q$ is a subgraph of $G_t$, and the other half, it is not.
      - Train on these examples by minimizing the following **max-margin loss**:
        - For positive examples: Minimize $E(G_q, G_t)$ when $G_q$ is a subgraph of $G_t$
        - For negative examples: Minimize $max(0, \alpha − E(G_q, G_t))$
        - Max-margin loss prevents the model from learning the degenerate strategy of moving embeddings further and further apart forever
    - Training Example Construction

      Need to generate training queries $G_Q$ and targets $G_T$ from the dataset $G$:
      - Get $G_T$ by choosing a random anchor $v$ and taking all nodes in $G$ within distance K from $v$ to be in $G_T$
      - Use **BFS sampling** to get $G_Q$. Sample induced subgraph of $G_T$:
        - Initialize $S = \{v\} , V = ∅$
        - Let $N(S)$ be all neighbors of nodes in $S$. At every step, sample 10% of the nodes in $N(S)/V$ and place them in $S$. Place the remaining nodes of $N(S)$ in $V$.
        - After K steps, take the subgraph of $G$ induced by $S$ anchored at $q$
      - For negative examples ($G_Q$ not subgraph of $G_T$), “corrupt” $G_Q$ by adding/removing nodes/edges so it’s no longer a subgraph

        <img src="src/L6/6.1.24.png" width="500">  

    - How many training examples to sample?
      - At every iteration, we sample new training pairs
      - Benefit: Every iteration, the model sees different subgraph examples
      - Improves performance and avoids overfitting – since there are exponential number of possible subgraphs to sample from
    - How deep is the BFS sampling?
      - A hyper-parameter that trades off runtime and performance
      - Usually use 3-5, depending on size of the dataset

- Subgraph Predictions on New Graphs

    Given: query graph $G_q$. anchored at node $q$, target graph $G_t$ anchored at node $t$

    Goal: output whether the query is a node anchored subgraph of the target

    Procedure: 
    - If $E(G_q, G_t)< \epsilon$, predict "True"; else "False"
    - $\epsilon$ a hyper-parameter

    To check if $G_Q$ is isomorphic to a subgraph of $G_T$, repeat this procedure for all $q ∈ G_Q$, $t ∈ G_T$. Here $G_q$ is the neighborhood around node $q ∈ G_Q$.

- Summary
  - Neural subgraph matching uses a machine learning based approach to learn the NP-hard problem of subgraph isomorphism
    - Given query and target graph, it embeds both graphs into an
order embedding space
    - Using these embeddings, it then computes $E(G_q, G_t)$ to
determine whether query is a subgraph of the target
  - Embedding graphs within an **order embedding space** allows subgraph isomorphism to be efficiently represented and tested by the relative positions of graph embeddings

#### 6.1.3 Finding Frequent Subgraphs

Finding the most frequent size-k motifs requires solving two challenges:
1. Enumerating all size-k connected subgraphs &rarr; Combinatorial explosion of number of possible patterns
2. Counting #(occurrences of each subgraph type) &rarr; Counting subgraph frequency is NP-hard

<img src="src/L6/6.1.25.png" width="300"> 

- Why is it Hard?

    - Just knowing if a certain subgraph exists in a graph is a hard computational problem! Subgraph isomorphism is NP-complete.

    - Computation time grows exponentially as the size of the subgraphs increases.
      - Feasible motif size for traditional methods is relatively small (3 to 7)

- Solution with Representation Learning
    - Combinatorial explosion &rarr; organize the search space

        Solution: Don't enumerate subgraphs but construct a size-k subgraph incrementally
        - Note: We are only interested in high frequency subgraphs
    - Subgraph isomorphism &rarr; prediction using GNN

        Solution: Use GNN to "predict" the frequency of the subgraph

- Problem Setup: Frequent Motif Mining
    - Target graph (dataset) $G_T$, size parameter $k$
    - Desired number of results $r$
    - Goal: Identify, among all possible graphs of $k$ nodes, the $r$ graphs with the highest frequency in $G_T$.
    - We use the node-level definition: The number of nodes $u$ in $G_T$ for which some subgraph of $G_T$ is isomorphic to $G_Q$ and the isomorphism maps $u$ to $v$ 

- SPMiner Overview
  - SPMiner: a neural model to identify frequent motifs
  - Key Idea: 
    - Decompose input graph $G_T$ into neighborhoods
    - Embed neighborhoods into an order embedding space
    - Key benefit of order embedding: We can quickly find out the frequency of a given subgraph $G_Q$

    <img src="src/L6/6.1.26.png" width="500">  

- Motif Frequency Estimation

    - Given: Set of subgraphs (“node-anchored neighborhoods”) ) $G_N$$_i$ of $G_T$ (sampled randomly)
    - Key idea: Estimate frequency of $G_Q$ by counting the number of $G_N$$_i$ such that their embeddings $z_N$$_i$ satisfy $z_Q$ ≤ $z_N$$_i$
    - Consequence of the order embedding space property

    <img src="src/L6/6.1.27.png" width="500">  

- SPMiner Search Procedure

    - Initial step: Start by randomly picking a starting node $u$ in the target graph. Set $S = \{u\}$

        <img src="src/L6/6.1.28.png" width="500"> 

    - Iteratively: Grow a motif by iteratively choosing a neighbor of a node in $S$, and adding that node to $S$ We want to grow motifs to find larger motifs!

        <img src="src/L6/6.1.29.png" width="500">  

    - Termination: Upon reaching a desired motif size, take the subgraph of the target graph induced by $S$

        <img src="src/L6/6.1.30.png" width="500">   

    - How to pick which node to add at each step?

        - Total violation of a subgraph G: number of neighborhoods
that do not contain G.
            - The number of neighborhoods $G_N$$_i$ that do **not** satisfy $z_Q$ ≤ $z_N$$_i$
            - Minimizing total violation = maximizing frequency
        - Greedy strategy (heuristic): At every step, add the node that results in the **smallest total violation**

- Example Results 
  - Small Motifs

    - Ground truth: find most frequent 10 motifs in dataset by brute-force exact enumeration (expensive)
    - Question: Can the model identify frequent motifs?
    - Result: The model identifies 9 and 8 of the top 10 motifs, respectively.

    <img src="src/L6/6.1.31.png" width="500">

  - Experiments on Large Motifs
    - Question: how do the frequencies of the identified motif compare?
    - Result: SPMiner identifies motifs that appear 10-100x more frequently than the baselines

    <img src="src/L6/6.1.32.png" width="500"> 

- Summary
  - Subgraphs and motifs are important concepts that provide insights into the structure of graphs. Their counts can be used as features for nodes and graphs.
  - We covered neural approaches to prediction subgraph isomorphism relationship.
  - Order embeddings have desirable properties and can be used to encode subgraph relations
  - Neural embedding-guided search in order embedding space can enable ML model to identify motifs much more frequent than existing methods

### 6.2 Community Structure in Networks

#### 6.2.1 Community Detection in Networks

- How does information flow through the network?

    There are different links (“short” vs. “long”) in the network, through which information flows.

    <img src="src/L6/6.2.1.png" width="400">  

- Example: Flow of Job Information

    Mark Granovetter, part of his PhD in 1960s - "How do people find out about new jobs?"

    - People find the information **through personal contacts**. 
    
    - But: Contacts were often **acquaintances** rather than close friends

    &rarr; This is surprising: One would expect your friends to help you out more than casual acquaintances. Why is it that acquaintances are most helpful?

    - Granovetter's Answer: Two perspectives on frendships

      - **Structural**: Friendships span different parts of the network
      - **Interpersonal**: Friendship between two people is either strong or weak

      <img src="src/L6/6.2.2.png" width="300">  

    - Granovetter's Explanation: Granovetter makes a connection between the social and structural role of an edge.       
      1. Structure:
           - Structurally embedded (tightly-connected) edges are also socially strong
           - Long-range edges spanning different parts of the network are socially weak
      2. Information:
           - Long-range edges allow you to gather information from different parts of the network and get a job
           - Structurally embedded edges are heavily redundant in terms of information access

    - How community (tightly-connected cluster of nodes) forms?

        If two people in a network have a friend in common, then there is an increased likelihood they will become friends themselves.

        e.g. Which edge is more likely, a-b or a-c?

        <img src="src/L6/6.2.3.png" width="200"> 

      - Triadic Closure = High clustering coefficient  

        Reasons for triadic closure: If B and C have a friend A in common, then

        &rarr; B is more likely to meet C (since they both spend time with A)

        &rarr; B and C trust each other (since they have a friend in common)

        &rarr; A has incentive to bring B and C together (since it is hard for A to maintain two disjoint relationships)

      - Empirical study by Bearman and Moody: Teenage girls with low clustering coefficient are more likely to contemplate suicide

- Edge Strength in Real Data

    - For many years Granovetter’s theory was not tested. But, today we have large who-talks-to-whom graphs: Email, Messenger, Cell phones, Facebook

    - Edge Overlap: 

        <img src="src/L6/6.2.4.png" width="500">       

    - Onnela et al. 2007: 
      - Cell-phone network of 20% of EU country’s population
      - Edge weight (strength): # phone calls
      - Observation: Highly used links have high overlap!

        <img src="src/L6/6.2.5.png" width="300"> 

        True: the real data

        Permuted strengths: Keep the network structure but randomly reassign edge strengths

- Edge Removal Based on Strength: Low to high, or, High to low

    <img src="src/L6/6.2.6.png" width="500">

- Edge Removal Based on Overlap: Low to high, or, High to low

    <img src="src/L6/6.2.7.png" width="500"> 


#### 6.2.2 Network Communities

Granovetter’s theory suggests that networks are composed of **tightly connected sets of nodes (communities)**:

<img src="src/L6/6.2.8.png" width="400"> 

Network communities (or clusters, groups, modules): Sets of nodes with **lots of internal** connections and **few external** ones (to the rest of the network).

- Ideally such automatically detected clusters would then correspond to real groups.

    - Social Network - Zachary's Karate club network
      - Observed social ties & rivalries in a university karate club
      - During the study, conflicts led the group to split
      - Split could be explained by a minimum cut in the network
  
      <img src="src/L6/6.2.9.png" width="400">  

    - Micro-Markets in Sponsored Search
      - Find micro-markets by partitioning the “query-to-advertiser” graph in web search:
      - Nodes: advertisers and queries/keywords
      - Edges: Advertiser advertising on a keyword.

      <img src="src/L6/6.2.10.png" width="300"> 

    - NCAA Football Network: Can we identify node groups? (communities, modules, clusters)

      <img src="src/L6/6.2.11.png" width="500">

- Define: Modularity $Q$

    A measure of how well a network is partitioned into communities

    Given a partitioning of the network into groups disjoint $s ∈ S$:

    <img src="src/L6/6.2.12.png" width="400">

    - Null Model: Configuration Model

        Given real $G$ on $n$ nodes and $m$ edges, construct rewired network $G'$:
      - **Same degree distribution** but uniformly random connections
      - Consider $G'$ as a **multigraph** (multiple edges exist between nodes)
      - The expected number of edges between nodes i and j of degrees $k_i$ and $k_j$ equals: <img src="src/L6/6.2.13.png" width="100">

        - There are $2m$ directed edges (counting i->j and j->i) in total.
        - For each of $k_i$ out-going edges from node $i$, the chance of it
landing to node $j$ is $k_j/2m$, hence $k_ik_j/2m$.
        - The expected number of edges in (multigraph) G’:

            <img src="src/L6/6.2.14.png" width="400">

    - Under null model, both the degree distribution and the total number of edges are preserved.

        Notice: This model applies to both weighted and unweighted networks. For weighted networks we use the weighted degree (sum of the edge weights).

    - Modularity of partitioning S of graph G:

        <img src="src/L6/6.2.15.png" width="400"> 
        
        $A_{ij}=1$ if i&rarr;j, 0 otherwise (if G is weighted then $A_{ij}$ is the edge weight)

    - Modularity values take range [-1,1]:
        - It is positive if the number of edges within groups exceeds the expected number
        - Q greater than **0.3-0.7** means **significant community structure**

    - Equivalently modularity can be written as:

        <img src="src/L6/6.2.16.png" width="500">

        Idea: We can identify communities by maximizing modularity.
        
#### 6.2.3 Louvain Algorithm

- Overview
  - Greedy algorithm for community detection
  - $O(nlog(n))$ run time
  - Supports weighted graphs
  - Provides hierarchical communities
  - Widely utilized to study large networks because fast, rapid convergence, and high modularity output (i.e. better communities)

- High-level: Louvain algorithm **greedily maximizes modularity**
  - Each pass is made of 2 phases:
    - Phase 1: Modularity is **optimized** by allowing only local changes to node-communities memberships
    - Phase 2: The identified communities are **aggregated** into super-nodes to build a new network
    - Go to Phase 1. The passes are repeated iteratively until no increase of modularity is possible.

    <img src="src/L6/6.2.17.png" width="500">  

  - Louvain algorithm considers graphs as **weighted**
    - The original graph can be unweighted (i.e., edge weights are all 1)
    - As the communities get identified and aggregated into super-nodes, weighted graphs are created (weights count the number of edges in the original graph)
    - Weighted version of Modularity is applied
- Louvain Phase 1: Partitioning
  - Put each node in a graph into a distinct community (one node per community)
  - For each node $i$, the algorithm performs two calculations:
    - Compute the modularity delta ($ΔQ$) when putting node $i$ into the community of some neighbor $j$
    - Move $i$ to a community of node $j$ that yields the largest gain in $ΔQ$
  - Phase 1 runs until no movement yields a gain
    - This first phase stops when a local maxima of the modularity is attained, i.e., when no individual node move can improve the modularity.

  Note that the output of the algorithm depends on the order in which the nodes are considered.

  Research indicates that the ordering of the nodes does not have a significant influence on the overall modularity that is obtained.

  - Louvain: Modularity Gain $ΔQ$

    What is $ΔQ$ if we move node $i$ from community D to C?

    $ΔQ(D$ &rarr; $i$ &rarr; $C) = ΔQ(D$ &rarr; $i) + ΔQ(i$ &rarr; $C)$

    <img src="src/L6/6.2.18.png" width="500">  

  - Deriving $ΔQ(i$ &rarr; $C)$:
    - First derive modularity within C: $Q(C)$
      - $\sum_{in} = \sum_{i,j\isin C}A_{ij}...$ sum of link weights **between** nodes in C
      - $\sum_{tot} = \sum_{i\isin C}k_{i}...$ sum of **all** link weights of nodes in C

      <img src="src/L6/6.2.19.png" width="400"> 

      Then we have

      <img src="src/L6/6.2.20.png" width="400">  

      $Q(C)$ is large when most of the total links are within-community links

    - Further define:
      - $k_{i,in} = \sum_{j\isin C}A_{ij}+\sum_{j\isin C}A_{ji}...$ sum of link weights **between** nodes $i$ and $C$
      - $k_i...$ sum of **all** link weights (degree) of node $i$

      <img src="src/L6/6.2.21.png" width="400">

      <img src="src/L6/6.2.22.png" width="500"> 

    - In summary, $ΔQ(i$ &rarr; $C) = Q_{after}-Q_{before}$. $ΔQ(D$ &rarr; $i)$ can be derived similarly.

  - Phase 1 Summary
    
    Iterate until no node moves to a new community:
    - For each node $i \isin V$ currently in community $C$, compute the best community C′: 

      $C'=argmax_{C'}ΔQ(C$ &rarr; $i$ &rarr; $C)$

      If $ΔQ(C$ &rarr; $i$ &rarr; $C)>0$ then update the community:
        - $C$ &rarr; $C - \{i\}$
        - $C'$ &rarr; $C' + \{i\}$

- Louvain Phase 2: Restructuring

  The communities obtained in the first phase are contracted into **super-nodes**, and the network is created accordingly:
  - Super-nodes are connected if there is at least one edge between the nodes of the corresponding communities
  - The weight of the edge between the two supernodes is the sum of the weights from all edges between their corresponding communities

  - **Phase 1 is then run on the super-node network.**

  <img src="src/L6/6.2.23.png" width="500"> 

  <img src="src/L6/6.2.24.png" width="500">  

- Summary
  - Modularity:
    - Overall quality of the partitioning of a graph into communities
    - Used to determine the number of communities
  - Louvain modularity maximization:
    - Greedy strategy 
    - Great performance, scales to large networks


#### 6.2.4 Detecting Overlapping Communities: BigCLAM

- Examples of Overlapping Communities

  Facebook Ego-network: Can we identify social communities?

  <img src="src/L6/6.2.26.png" width="500"> 

  Protein-Protein Interactions: Can we identify functional modules?

  <img src="src/L6/6.2.27.png" width="500"> 


- Non-overlapping vs. overlapping communities

  <img src="src/L6/6.2.25.png" width="500"> 

- Action Plan

  1. Define a generative model for graphs that is based on node community affiliations:
      - Community-Affiliation Graph Model (AGM)

  2. Given Graph $G$, make the assumption that $G$ was generated by AGM. Find the best AGM that could have generated $G$

      &rarr; this way we discover communities

- Community-Affiliation Graph Model (AGM)

  Generative model: How is a network generated from community affiliations?

  - Model Parameters:
    - Nodes V, Communities C, Memberships M
    - Each community $c$ has a single probability $p_c$

  <img src="src/L6/6.2.28.png" width="500"> 

  - AGM Generative Process:

    Given parameters $(V, C, M, \{p_c\})$:
    - Nodes in community c connect to each other by flipping a coin with probability $p_c$
    - Nodes that belong to multiple communities have multiple coin flips

      If they “miss” the first time, they get another chance through the next community

      <img src="src/L6/6.2.29.png" width="400"> 

      Note: If nodes u and v have no communities in common, then p(u,v)=0. We resolve this by having a background “epsilon” community that every node is a member of.

  - AGM Flexibility
    - AGM can express a variety of community structures: Non-overlapping, Overlapping, Nested

    <img src="src/L6/6.2.30.png" width="500"> 

- Detecting Communities with AGM

  Given a Graph, find the model $F$
  1) Affiliation graph $M$
  2) Number of communities $C$
  3) Parameters $p_c$

  - How to estimate model parameters $F$ given a $G$?

    Solution: **Maximum likelihood estimation**

    Find model/parameters $F$ which <img src="src/L6/6.2.31.png" width="300">

    To solve this we need to:
    - Efficiently calculate $P(G|F)$
    - Then maximize over $F$ (e.g., using gradient descent)
  - Graph Likelihood $P(G|F)$: likelihood that $F$ generated $G$

    <img src="src/L6/6.2.32.png" width="500"> 

  - "Relaxing" AGM towards $P(u,v)$

    Memberships have strengths:  

    - $F_{uA}$: The membership strength of node $u$ to community $A$ (0 means no membership)

    <img src="src/L6/6.2.33.png" width="300"> 
    <img src="src/L6/6.2.34.png" width="130">

    For community $C$, we model the probability of $u$ and $v$ being connected as: 
    
    - $P_C(u,v)=1-exp(-F_{uC}\cdot F_{vC})$, where $F$ is non-negative membership strength.

      $P_C(u,v)$ staisfies $0<= P_C(u,v)<=1$ (valid probability) because $F_{uC}\cdot F_{vC}>=0$
      - $P_C(u,v)=0$ iff $F_{uC}\cdot F_{vC}=0$ i.e. $F_{uC}=0$ or $F_{vC}=0$. 
    
        &rarr; That is, Nodes $u$ or $v$ are not connected via $C$ iff at least one of them has zero membership strength for $C$.
    
      - $P_C(u,v) \approx 1$ iff $F_{uC}\cdot F_{vC}$ is large

        &rarr; Nodes $u$ or $v$ are connected via $C$ iff both have high membership strength for $C$
    
    Generalization: Nodes $u$ and $v$ can connect via multiple communities.
    
    <img src="src/L6/6.2.35.png" width="300">  

    - Probability that $u$ and $v$ are connected by **at least one of the communities**:

      <img src="src/L6/6.2.36.png" width="500">

      <img src="src/L6/6.2.37.png" width="500">

      &rarr; **Prob. of nodes $u$ and $v$ linking is proportional to the strength of shared memberships**

- BigCLAM Model

  - Optimization Objective
    
    Given a network $G(V, E)$, we maximize the likelihood (probability) of $G$ under our model:

    <img src="src/L6/6.2.38.png" width="500">

    Likelihood involves a product of many small probabilities &rarr; Numerically unstable. Instead, we consider the log likelihood:

    <img src="src/L6/6.2.39.png" width="500"> 

  - Optimizing $l(F)$
    - Start with random membership $F$
    - Iterate until convergence:

      For $u \isin V$,
      - Update membership $F_u$ for node $u$ while fixing the memberships of all other nodes
      - Specifically, we do gradient ascent, where we make small changes to $F_u$ that lead to increase in log-likelihood.

        Partial derivative w.r.t $F_u$: <img src="src/L6/6.2.40.png" width="400">

  - Time complexity of gradient ascent:

    <img src="src/L6/6.2.41.png" width="400"> 


    Naïve gradient ascent is slow! However:

    <img src="src/L6/6.2.42.png" width="300"> 

    - $\sum_vF_v$ can be computed at the beginning and efficiently updated whenever $F_u$ is updated.
    - The gradient step takes **linear time** in the degree of $u$!
  - Summary
    - BigCLAM defines the model to generate a network with **overlapping community structure**.
    - Given a graph, BigCLAM’s parameters (**membership strength of each node**) can be estimated by maximizing the log-likelihood of generating the graph under the model.