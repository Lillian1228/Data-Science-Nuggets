## Classification Loss
### 1. Multi-class Cross-Entropy Loss 

Cross-entropy loss, also known as log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. For multi-class classification tasks, we use the categorical cross-entropy loss.

#### Mathematical Background

For a single sample with C classes, the categorical cross-entropy loss is defined as:

$L = -\sum_{c=1}^{C} y_c \log(p_c)$

where:

- $y_c$ is a binary indicator (0 or 1) if class label c is the correct classification for the sample
- $p_c$ is the predicted probability that the sample belongs to class c
- $C$ is the number of classes

#### Implementation Requirements

Your task is to implement a function that computes the average cross-entropy loss across multiple samples:

$L_{batch} = -\frac{1}{N}\sum_{n=1}^{N}\sum_{c=1}^{C} y_{n,c} \log(p_{n,c})$

where N is the number of samples in the batch.

#### Important Considerations

- Handle numerical stability by adding a small epsilon to avoid log(0)
    - if any predicted probability $p_{ij}$ is exactly 0, then $log(p_{ij})$ &rarr;$-\inf$
    - Fix: we clip the predicted probabilities away from 0 (and 1 sometimes too): ```np.clip(P, eps, 1-eps)```
- Ensure predicted probabilities sum to 1 for each sample
- Return average loss across all samples
- Handle invalid inputs appropriately

The function should take predicted probabilities and true labels as input and return the average cross-entropy loss.

```Python
def cross_entropy_loss(Y, P, eps=1e-15):
    # Clip predictions to avoid log(0)
    P = np.clip(P, eps, 1 - eps)
    m = Y.shape[0]
    loss = -(1/m) * np.sum(Y * np.log(P)) # element-wise multiplication
    return loss
```

### 2. Focal Loss (For Imbalanced Classification)

#### **2.1 Motivation**

* Standard **cross-entropy loss** treats all samples equally:

$$
L_{\text{CE}} = - \sum_{i=1}^m \sum_{c=1}^K y_{ic} \log p_{ic}
$$

* In **imbalanced datasets**, majority class dominates the gradient. Minority class samples contribute very little, so the model may ignore them.

* **Focal Loss** addresses this by **down-weighting easy examples** and focusing training on **hard, misclassified examples**.

#### **2.2 Multi-class Focal Loss formula**

Let:

* $y_{ic} \in \{0,1\}$ be the one-hot true label for sample $i$, class $c$
* $p_{ic}$ be predicted probability for class $c$
* $\gamma \ge 0$ be the **focusing parameter** (controls down-weighting of easy examples)
* $\alpha_c \in [0,1]$ optional **class weighting** for handling imbalance

$$
\boxed{
L_{\text{focal}} = - \frac{1}{m} \sum_{i=1}^m \sum_{c=1}^K \alpha_c \, y_{ic} \, (1 - p_{ic})^\gamma \, \log p_{ic}
}
$$

* When $\gamma = 0$, Focal Loss reduces to weighted cross-entropy.
* $(1 - p_{ic})^\gamma$ **down-weights well-classified examples** (large $p_{ic}$ → small contribution).

#### **2.3 Why better for imbalanced classification**

* In imbalanced datasets:

  * Majority class often gets predicted with high confidence → easy examples
  * Cross-entropy gradient dominated by majority class

* Focal Loss reduces the weight of **easy examples** → minority / hard examples have relatively larger gradients → model learns them better.

* The $\alpha_c$ term can also **balance class importance** explicitly.

Practical rule of thumb:

| Minority\:Majority Ratio | Imbalance Level | Recommendation                                | Notes                                                                            |
| ------------------------ | --------------- | --------------------------------------------- | -------------------------------------------------------------------------------- |
| 1:1 – 1:2                | Slight          | Standard cross-entropy                        | No special loss needed.                                                          |
| 1:3 – 1:5                | Moderate        | Weighted cross-entropy; optionally focal loss | Weighting is usually sufficient; focal loss helps if minority examples are hard. |
| 1:6 – 1:10               | Moderate-High   | Focal loss or weighted CE                     | Focal loss helps focus on minority class and hard examples.                      |
| 1:10+                    | High / Severe   | Focal loss (possibly with alpha weighting)    | Standard CE often fails; focal loss strongly recommended.                        |


#### 2.4 Notes on implementation

* Typical $\gamma = 2$ works well.

* Optional $\alpha_c$ can be set as inverse frequency of each class.

* Works with **softmax outputs** in multi-class classification.

* Python/NumPy pseudocode:

```python
import numpy as np

def focal_loss(Y, P, gamma=2.0, alpha=None, eps=1e-8):
    P = np.clip(P, eps, 1 - eps)
    m = Y.shape[0]
    if alpha is None:
        alpha = np.ones(Y.shape[1]) # (k,)
    alpha = alpha.reshape(1, -1)  # broadcast to (1,k): set the first dim as 1 and let numpy infer the other
    loss = -np.sum(alpha * Y * ((1 - P) ** gamma) * np.log(P)) / m
    return loss
```

Step by step:

1. 1 - P → shape (m, K)

2. (1 - P) ** gamma → shape (m, K) (elementwise exponentiation)

3. np.log(P) → shape (m, K)

4. Y * ((1 - P) ** gamma) * np.log(P) → shape (m, K) (elementwise multiplication)

5. alpha * ... → alpha shape (1,K) broadcasts across m rows → result shape (m,K)

6. np.sum(...) sums over all m*K elements → scalar.


✅ **Takeaways:**

* Focal loss = **weighted, focusing cross-entropy**
* Helps when dataset is **imbalanced or many easy examples**
* Reduces gradient contribution from correctly classified (easy) samples
* Use $\gamma > 0$ and optionally $\alpha_c$ for class imbalance

### 3. Other Classification Loss

* **Hinge Loss:** (used in SVMs)

$$
L = \sum_i \max(0, 1 - y_i \hat y_i)
$$

* **Squared Hinge Loss:**

$$
L = \sum_i (\max(0, 1 - y_i \hat y_i))^2
$$

* **Kullback-Leibler (KL) Divergence:**

$$
D_{\text{KL}}(P\|Q) = \sum_i P_i \log \frac{P_i}{Q_i}
$$

→ Often used for probabilistic outputs.

## Regression Loss

- Mean Squared Error (MSE):

$$
\text{MSE} = \frac{1}{m} \sum_{i=1}^m (y_i - \hat y_i)^2
$$

* **Mean Absolute Error (MAE):**

$$
\text{MAE} = \frac{1}{m} \sum_{i=1}^m |y_i - \hat y_i|
$$

* **Huber Loss:**

    $$
    L_\delta(y_i, \hat y_i) = 
    \begin{cases} 
    \frac{1}{2}(y_i - \hat y_i)^2 & \text{if } |y_i - \hat y_i| \le \delta\\
    \delta \, |y_i - \hat y_i| - \frac{1}{2} \delta^2 & \text{otherwise}
    \end{cases}
    $$

    → Combines MSE and MAE; robust to outliers.

* **Log-Cosh Loss:**

    $$
    L(y_i, \hat y_i) = \log(\cosh(\hat y_i - y_i))
    $$

    → Smooth approximation of MAE.


## Ranking / Pairwise Losses

* **Pairwise Hinge Loss:** for ranking (used in learning-to-rank).
* **Triplet Loss:** used in embeddings (face recognition, metric learning).

$$
L = \max(0, d(a,p) - d(a,n) + \text{margin})
$$

---

## Other / Specialized Losses

* **Cosine Similarity Loss:** for embeddings, representation learning.

$$
L = 1 - \frac{\langle y, \hat y \rangle}{\|y\|\|\hat y\|}
$$

* **Dice Loss / Jaccard Loss:** for segmentation tasks in images.

$$
L_{\text{Dice}} = 1 - \frac{2 |Y \cap \hat Y|}{|Y| + |\hat Y|}
$$

* **Poisson Loss:** for count data.

