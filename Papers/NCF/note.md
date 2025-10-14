*He et al., “Neural Collaborative Filtering,” WWW 2017*

### 🧭 Motivation

In ranking tasks, the goal is to estimate the user-item interactions from implicit feedbacks like user clicks.

Matrix Factorization (MF) dominated collaborative filtering but relies on a **fixed, linear inner-product interaction** between user and item latent factors.

- How it works

    Matrix Factorization assumes that a user’s preference for an item can be **explained by interactions in a shared latent space**.

    Formally, given a **user–item interaction matrix** $R \in \mathbb{R}^{m \times n}$:

    * $m$: number of users
    * $n$: number of items
    * $R_{ui}$: observed interaction (e.g., rating, click, purchase)

    We aim to approximate $R$ as: $$R \approx P Q^T$$
    

    where

    * $P \in \mathbb{R}^{m \times k}$: user latent matrix
    * $Q \in \mathbb{R}^{n \times k}$: item latent matrix
    * $k$: latent dimension (usually 10–200)

    Each:

    * $p_u$: latent vector for user $u$ (row of $P$)
    * $q_i$: latent vector for item $i$ (row of $Q$)

    The **predicted preference score** (interaction strength) is the **inner product** of these two latent vectors:


    $$\hat{r}*{ui} = p_u^T q_i = \sum*{f=1}^{k} p_{uf} q_{if}$$


    This dot product measures **alignment (similarity)** in latent space:

    * If $p_u$ and $q_i$ point in similar directions → high score (strong preference)
    * If they are orthogonal or opposite → low or negative score (weak preference)

* That linearity cannot capture **complex, nonlinear relationships** in implicit-feedback data (clicks, purchases, views).
* Past “deep” recommenders mainly used DNNs to model *side information* (text, audio, images) yet still kept MF’s inner product for user–item matching.
* Authors aimed to let a neural network **learn the interaction function f(u,i)** directly from data, unifying MF and deep learning in one framework for implicit feedback.


### 🧮 Data Preparation & Experimental Setup

**Datasets**

| Dataset      | Type                                               | #Users | #Items | #Interactions | Sparsity |
| ------------ | -------------------------------------------------- | ------ | ------ | ------------- | -------- |
| MovieLens 1M | Explicit ratings converted to implicit (1 = rated) | 6 040  | 3 706  | 1 000 209     | 95.5 %   |
| Pinterest    | Implicit (pin action = 1)                          | 55 187 | 9 916  | 1 500 809     | 99.7 %   |

* Users with < 20 interactions removed.
* **Leave-one-out evaluation:** each user’s latest interaction → test item; remaining → train.
* Each test item ranked among 100 candidates (1 positive + 99 negatives).
* **Metrics:** Hit Ratio (HR@10) and NDCG@10.
* **Negative sampling:** 4 negatives per positive (default).
* **Optimization:** log loss (binary cross-entropy) via mini-batch Adam; later SGD for fine-tuning NeuMF.
* Hyper-parameters tuned on validation; embedding sizes 8–64; batch 128–1024.


### 🧠 Model Architecture

### 1️⃣ General Framework (NCF)

Predicts $\hat y_{ui}=f(P^Tv^U_u,Q^Tv^I_i|\Theta_f)$ using a multi-layer network.

* Input layer: user and item IDs (one-hot vectors).
* Embedding layer: dense latent vectors (p_u, q_i).

    NCF adopts two pathways to model users and items, and combine the features of two pathways by concatenating them. This design has been widely adopted in multimodal deep learning work.

* Neural CF layers: stack of hidden Multi-Layer Perceptron (MLP) layers that learn interaction function f.

    However, simply a vector concatenation does not account for any interactions between user and item latent features, which is insufficient for modelling the collaborative  ltering e ect. To address this issue, we propose to add hidden layers on the concatenated vector, using a standard MLP to learn the interaction between user and item latent features.

    * Concatenates (p_u) and (q_i): ([p_u; q_i]).
    * Passes through multiple non-linear hidden layers (ReLU).
    * Learns complex user–item interactions beyond fixed element-wise operations.
    * Tower structure: each higher layer has ½ neurons of previous.

* Output layer: probability that user u interacts with item i (logistic activation).
* Loss: binary cross-entropy with negative sampling.

### 2️⃣ Generalized Matrix Factorization (GMF)

* Element-wise product (p_u ⊙ q_i) followed by learned weights and sigmoid.
* Reduces to MF if weights = 1 and identity activation.
* Introduces trainable weights h and nonlinearity σ to make MF more flexible.


### 4️⃣ Neural Matrix Factorization (NeuMF)

* **Fusion** of GMF (linear) and MLP (non-linear).
* GMF and MLP have separate embeddings; their final latent outputs concatenated → sigmoid output.
* Combines MF’s memorization and MLP’s generalization.

**Pre-training:**
Train GMF and MLP individually, then use their weights to initialize NeuMF (helps avoid poor local minima).
Final output weights (h =$$α h_{GMF}, (1–α) h_{MLP}$$) with α ≈ 0.5.

---

## 📊 Results & Empirical Findings

### RQ1 – Do NCF methods outperform baselines?

**Baselines:** ItemPop, ItemKNN, BPR, eALS.
**Findings:**

* **NeuMF wins on all datasets** and metrics (HR@10, NDCG@10).
   Relative gain ≈ +4–5 % over eALS/BPR.
* **GMF > BPR** → log-loss superior to pairwise BPR loss for implicit data.
* **MLP ≈ GMF** (MLP slightly lower but improves with depth).
* ItemPop worst → personalization is critical.

---

### RQ2 – Effect of Log Loss & Negative Sampling

* Training loss monotonically ↓ first 10 epochs; later overfits.
* NeuMF has lowest training loss and best HR/NDCG.
* Optimal # negatives ≈ 3–6 per positive; too few hurts learning, too many → noise.
* Demonstrates pointwise log loss is flexible and more robust than pairwise objectives.

---

### RQ3 – Is Deep Learning Helpful?

* MLP performance increases monotonically with depth:
   MLP-0 (no hidden layer) ≪ MLP-4.
* ReLU > tanh > sigmoid.
* Stacking non-linear layers captures higher-order user–item patterns.
* Linear layers alone provide no gain → non-linearity is key.

---

### Utility of Pre-training

* Pre-trained NeuMF > random init NeuMF (≈ +1–2 % HR/NDCG improvement).
* Helps stabilize convergence in non-convex optimization.

---

## ⚙️ Limitations and Future Work

**Limitations**

* Evaluated only on implicit binary data; no temporal or contextual signals.
* Used pointwise log loss only; pairwise ranking objective left unexplored.
* Small datasets (MovieLens 1M, Pinterest subset) → may not reflect web-scale behavior.
* No integration with side features (content, reviews, multimedia).

**Future directions**

* Extend to **pairwise learners** and **auxiliary information** (user reviews, knowledge bases, temporal signals).
* Explore **group recommendation**, **multi-modal content** (images, videos), and **multi-view representation learning**.
* Investigate **recurrent networks** and **hashing** for efficient online retrieval.

---

## 🧾 Summary at a Glance

| Aspect              | Details                                                                                           |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| **Goal**            | Replace MF’s fixed inner product with a neural interaction function for implicit CF.              |
| **Key Models**      | GMF (linear), MLP (non-linear), NeuMF (hybrid fusion).                                            |
| **Training Signal** | Implicit feedback → binary classification (log loss + negative sampling).                         |
| **Data**            | MovieLens 1M and Pinterest implicit datasets.                                                     |
| **Results**         | NeuMF achieves ≈ 5 % relative gain over state-of-the-art MF (BPR, eALS).                          |
| **Takeaway**        | Deep non-linear layers enhance CF expressiveness; pointwise log loss effective for implicit data. |
| **Future Work**     | Pairwise learners, context/temporal extensions, multi-modal recsys, online RNN retrieval.         |

---

Would you like a **diagram showing GMF + MLP → NeuMF architecture** annotated with key equations and training flow (embedding, fusion, loss)? It can serve as a concise study sheet for your notes.



---

## 📉 3. Training Objective

MF learns $p_u, q_i$ by minimizing the reconstruction error (or ranking loss), depending on the task.

### (a) For explicit ratings:

[
\min_{P, Q} \sum_{(u,i) \in \mathcal{K}} (R_{ui} - p_u^T q_i)^2 + \lambda (|p_u|^2 + |q_i|^2)
]

where $\mathcal{K}$ = observed user–item pairs, and $\lambda$ is regularization.

### (b) For implicit feedback:

Use confidence-weighted loss (e.g., *Weighted Matrix Factorization* / *ALS*):
[
\min_{P, Q} \sum_{u,i} c_{ui}(p_u^T q_i - s_{ui})^2
]
where:

* $s_{ui} = 1$ if interaction observed else 0
* $c_{ui} = 1 + \alpha r_{ui}$: confidence level

or pairwise (BPR):
[
\min_{P,Q} -\sum_{(u,i,j)} \log \sigma(p_u^T q_i - p_u^T q_j)
]
where $i$ is a positive and $j$ a negative item for user $u$.

---

## 📊 4. Intuitive Interpretation

You can think of:

* **User vector $p_u$** as what the user “cares about” (latent preferences).
* **Item vector $q_i$** as what the item “offers” (latent attributes).
* Their **dot product** captures the *compatibility* or *match strength* between user and item.

> Example:
>
> * A movie’s latent dimensions might capture "action", "romance", "humor".
> * A user vector might weight those as preferences.
> * Their dot product ≈ how much that movie fits the user’s taste.

---

## 🧩 5. Limitations That Led to Neural CF

| Limitation of MF                           | Neural CF Improvement                            |
| ------------------------------------------ | ------------------------------------------------ |
| Assumes linear interaction (inner product) | Learns nonlinear interactions via neural layers  |
| Hard to incorporate complex features       | Easily integrates embeddings for content/context |
| Struggles with implicit sparse data        | NCF designed specifically for implicit feedback  |
| Cold-start problem persists                | Extendable with feature-based encoders           |

---

✅ **Summary**

> Matrix Factorization estimates a user–item interaction as the **inner product of two latent vectors** learned to reconstruct observed feedback.
> It’s linear, efficient, and interpretable — but limited in flexibility, which motivated Neural Collaborative Filtering (NCF) to replace the inner product with a learnable neural interaction function.
