## Introduction

Decision forests are a family of interpretable machine learning algorithms that excel with tabular data. Decision forests can perform:

- Classification
- Regression
- Ranking
- Anomaly detection
- Uplift modeling.

Decision forests provide the following benefits:

- They are **easier to configure** than neural networks. Decision forests have **fewer hyperparameters**; furthermore, the hyperparameters in decision forests provide **good defaults**.
- They **natively handle** numeric, categorical, and missing features. This means you can write far less preprocessing code than when using a neural network, saving you time and reducing sources for error.
- They often give **good results out of the box**, are robust to noisy data, and have interpretable properties.
- They infer and train on **small datasets (< 1M examples) much faster** than neural networks.

&rarr; Decision forests are highly effective for modeling tabular data, making them a primary choice for datasets commonly found in spreadsheets, CSV files, or databases.

## Decision Trees

Decision forest models are composed of decision trees. Decision forest learning algorithms (like random forests) rely, at least in part, on the learning of decision trees.

A **decision tree** model uses a hierarchical structure of conditions to route an example from the root to a leaf node, where the leaf's value represents the prediction. The predictions can be both a specific class or numeric values (regression).

<img src="src/9.png" width="400">

#### Types of conditions

- An **axis-aligned condition** involves only a single feature. i.e. num_legs>=2
- An **oblique condition** involves multiple features. i.e. num_legs ≥ num_fingers

    While oblique splits offer greater power, they come at the cost of increased training and inference expenses compared to axis-aligned conditions.
- Conditions can be **binary (two outcomes)** or **non-binary** (more than two outcomes), with binary decision trees being more common due to reduced overfitting.
- **Threshold conditions**, comparing a feature to a threshold, are the most frequently used type of condition in decision trees.

## Growing decision trees

### General Idea

The optimal training of a decision tree is an NP-hard problem. Therefore, training is generally done using heuristics. 

Most algorithms used to train decision trees work with a **greedy divide and conquer strategy**. The algorithm starts by creating a single node (the root) and recursively grows the tree by selecting the "best" condition at each node.

At each node, all the possible conditions are evaluated and scored. The algorithm selects the "best" condition, that is, the condition with the highest score. This routine is called the **splitter**.
- The choice of splitter algorithm depends on factors such as feature type, task, condition type, and regularization criteria.
- The score maximized by the splitter depends on the task. For example:
    - **Information gain** and **Gini** are commonly used for classification.
    - **Mean squared error** is commonly used for regression.

When no satisfying conditions are found, the node becomes a leaf. The leaf prediction is determined as the most representative label value in the examples.

### Exact splitter for binary classification

The simplest splitter algorithm for decision trees uses a threshold on a numerical feature to divide data into two groups, aiming to improve label separation.

**Information gain**, based on Shannon entropy, quantifies the improvement in label separation achieved by a split, with higher values indicating better splits.

- **Entropy is a measure of disorder**. For a binary label, high entropy means labels in the examples are quite balanced, and a low entropy means the labels in the examples are pure.

    $H(X) = -p \cdot log p - q \cdot log q = -p \cdot log p - (1-p) log (1-p)$, where $p$ is the ratio of positive label values in examples $X$
    
    <img src="src/10.png" width="400">  

- $IG(D,T,F)$ is the information gain of splitting D into T and F, that is, the entropy delta between $D$ and $\{T,F\}$ weighted by the number of examples.

    <img src="src/10_1.png" width="300">  

Example - threshold condition of a single numerical feature $x$

<img src="src/10_2.png" width="500"> 

- The algorithm efficiently finds the optimal threshold by sorting feature values and testing midpoints between consecutive values, with a time complexity of O(n log n).

- Decision tree training with this splitter, applied to each node and feature, has a time complexity of O(m n log² n), where 'm' is the number of features and 'n' is the number of training examples.

- This algorithm is insensitive to the scale or distribution of feature values, eliminating the need for feature normalization or scaling before training.

Code example: Compute the information gain of the following split

|Node	| # of positive examples	| # of negative examples |
|-----|----|----|
|parent node |	10	| 6
|first child |	8	| 2
|second child |	2	| 4


```
# Positive label distribution
p_parent = 10 / (10+6) # = 0.625
p_child_1 = 8 / (8+2) # = 0.8
p_child_2 = 2 / (2+4) # = 0.3333333

# Entropy
h_parent = -p_parent * log(p_parent) - (1-p_parent) * log(1-p_parent) # = 0.6615632
h_child_1 = ... # = 0.5004024
h_child_2 = ... # = 0.6365142

# Ratio of example in the child 1
s = (8+2)/(10+6)
f_final = s * h_child_1 + (1-s) * h_child_2 # = 0.5514443

information_gain = h_parent - f_final # = 0.1101189
```

### Overfitting and regularization

To limit overfitting a decision tree, apply one or all of the following regularization criteria while training the decision tree:

- Set a **maximum depth**: Prevent decision trees from growing past a maximum depth, such as 10.
- Set a **minimum number of examples in leaf**: A leaf with less than a certain number of examples will not be considered for splitting.
- **Prune** the decision tree: selectively removing (**pruning**) certain branches. A common solution is to use a validation dataset. If removing a branch improves the quality of the model on the validation dataset, then the branch is removed.

### Variable importance

Because different variable importances inform about different aspects of the models, looking at several variable importances at the same time is informative.

- Model-agnostic **permutation variable importances**: evaluates the increase in the prediction error of a model after permuting the feature's values.
- Decision tree specific
  - sum of the split score with a given variable
  - number of nodes with a given variable
  - average depth of the first occurrence of a feature across all the tree paths.

If a feature is important according to all the variable importances, this feature is likely important. 

If a feature has a high "number of nodes" variable importance and a small "permutation" variable importance, then this feature might be hard to generalize and can hurt the model quality. i.e. this can happen when a feature is just an example identifier with no information to generalize.