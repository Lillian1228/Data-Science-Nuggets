### 4.1 Motivating Node Classification

Main question: Given a network with labels on some nodes, how do we assign labels to all other nodes in the network?

Example: In a network, some nodes are fraudsters, and some other nodes are fully trusted. How do you find the other fraudsters and trustworthy nodes?

Node embeddings (from random walks, GNN) is a method to solve this problem. Are there **alternative solutions based on the network topology**?

Problem Setting: Transductive node classification (also called semi-supervised node classification) - Given labels of some nodes, predict labels of unlabeled nodes.

<img src="src/L4/4.1.1.png" width="300">  

**Label propagation** is an alternative framework. Intuition is that correlations exist in networks. Connected nodes tend to share the same label.

#### Corelations in Networks

Observation: Behaviors of nodes are correlated across the links of the network. i.e. Nearby nodes have the same color (belonging to the same class).

Two explanations for why behaviors of nodes in networks are correlated:

<img src="src/L4/4.1.2.png" width="300">  

- Homophily: The tendency of individuals to associate and bond with similar others.
  - “Birds of a feather flock together”
  - It has been observed in a vast array of network studies, based on a variety of
attributes (e.g., age, gender, organizational role, etc.)
  - Example: online social network, where people with the same interest are more closely connected due to homophily.
    - nodes=people
    - edge=frendship
    - node color=interests (sports, arts, etc.) 
- Influence: Social connections can influence the individual characteristics of a person.
  - Example: I recommend my musical preferences to my friends, until one of them grows to like my same favorite genres!

#### How do we leverage node correlations in networks?

Semi-supervised Learning: Given graph with few labeled nodes, find class of remaining nodes.
- Main assumption: There is homophily in the network, that is, connected nodes tend to share the same label.
- Example task:
  - Let 𝑨 be a 𝑛×𝑛 adjacency matrix over 𝑛 nodes
  - Let $Y=[0,1]^𝑛$ be a vector of labels: $Y_v$=1 belongs to Class 1; $Y_v$=0 belongs to Class 0. There are unlabeled node needs to be classified.
  - Each node $v$ has a feature vector $f_v$
  - Task: Predict $P(Y_v)$ given all features and the network

- We will look at 3 techniques today:
  - Label propagation
  - Correct & Smooth
  - Masked label prediction

### 4.2 Label Propagation

Idea: Propagate node labels across the network. Class probability of node $v$, $P(Y_v)$, is a weighted average of class probabilities of its neighbors.
- For labeled nodes 𝑣, initialize label $𝑌_𝑣$ with ground truth label $𝑌_𝑣^*$
- For unlabeled nodes , initialize $𝑌_𝑣$=0.5
- Update each unlabeled node v's label. The math:

    <img src="src/L4/4.1.3.png" width="400">   

    If edges have strength/weight information, $𝐴_{𝑣,𝑢}$ can be the edge weight between 𝑣 and 𝑢.

    $P(Y_v=c)$ is the probability of node 𝑣having label 𝑐

- Repeated the update until convergence or until maximum number of iterations is reached.
  
  Intuition of convergence: Stop iterating when all nodes have reached a static value. Applying label propagation does not change labels anymore.

  Convergence criteria: $|P^{(t)}(Y_v)-P^{(t-1)}(Y_v)|\leq𝜖$ for all nodes $v$. Here 𝜖 is the convergence threshold.

Example: Initialization

<img src="src/L4/4.1.4.png" width="500"> 

- Update for the 1st Iteration:
    
    Node 3: $N_3$ = {1,2,4} &rarr; $𝑃(𝑌_3)=(0+0+0.5)/3=0.17$

    Node 4: $N_4$ = {1,3,5,6} &rarr; $𝑃(𝑌_4)=(0+0.5+0.5+1)/4=0.5$

    Note that we update based on the scores in the previous iteration. After Iteration 1 of updating all the unlabeled nodes:

    <img src="src/L4/4.1.5.png" width="500">  

- After 2nd iteration

    <img src="src/L4/4.1.6.png" width="500">  

- After 3rd iteration

    <img src="src/L4/4.1.7.png" width="500">  

- After 4th iteration

    <img src="src/L4/4.1.8.png" width="500">  

- Convergence: All scores stabilize after 4 iterations. We therefore predict
  - Node 4, 5, 8, 9 belongs to class 1 ($P(Y_v)$>0.5)
  - Node 3 belong to class 0.

  <img src="src/L4/4.1.9.png" width="500">  

#### Label Propagation Applications
- Document classification
  - Nodes: documents
  - Edges: document similarity (e.g. word overlap)
- Twitter polarity classification
  - Nodes: users, tweets, words, hashtags, emoji
  - Edges [users &rarr; users] Twitter follower graph;
  - Edges [users &rarr; tweets] creation;
  - Edges [tweets &rarr; words/hashtags/emoji]
- Spam and fraud detection

#### Label Propagation Summary

- Method: Iteratively update probabilities of node belonging to a label class based on its neighbors
- Issues:
  - Convergence may be very slow and not guaranteed
  - does not use node attributes
- Can we improve the idea of label propagation to leverage attribute/feature information?

### 4.3 Node Classification: Correct & Smooth

Correct & Smooth (C&S) is a recent state of the art node classification method. It tops the [OGB leaderboard](https://ogb.stanford.edu/docs/leader_nodeprop/)!

C&S Motivations:

