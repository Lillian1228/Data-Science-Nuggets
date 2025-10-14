
## 1. Rec System Architecture Ovreview

A large-scale recommendation system typically follows a **multi-stage architecture**, optimized for both **accuracy** and **latency**. It usually consists of **three main layers**:

1. **Candidate Generation (Retrieval)**
2. **Scoring / Ranking**: can be further split to pre-rank + rank
3. **Re-ranking / Post-ranking**

These stages sit on top of strong **data and infrastructure layers** for feature engineering, model serving, and feedback loops.

```
User → Logging → Feature Store → Candidate Generation → Ranking → Re-ranking → UI Delivery
```

**Components:**

| Layer                                | Goal                                                                    | Typical Models / Techniques                                             |
| ------------------------------------ | ----------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **Data & Logging**                   | Collect user-item interactions (clicks, views, dwell time, likes, etc.) | Event pipelines (Kafka, Flink, Spark Streaming)                         |
| **Feature Store**                    | Store precomputed user/item/context features                            | Feast, Vertex AI Feature Store, custom pipelines                        |
| **Candidate Generation (Retrieval)** | Narrow billions of items to hundreds                                    | Two-tower (Siamese) networks, Approximate Nearest Neighbor (ANN) search |
| **Ranking**                          | Score and sort candidates by relevance                                  | DNNs, Gradient Boosted Trees, Wide & Deep, Transformer-based rankers    |
| **Re-ranking / Post-processing**     | Enforce diversity, freshness, fairness, business rules                  | Multi-objective optimization, MMR, bandits                              |
| **Serving**                          | Real-time inference, caching                                            | TensorFlow Serving, PyTorch Serve, Faiss/ScaNN for ANN                  |
| **Feedback Loop**                    | Collect results, retrain models                                         | Airflow/SageMaker pipelines, online learning                            |


### A/B Testing




## 2. Retrieval Stage

### 2.1 ItemCF: Item Based Collaborative Filtering

#### 💡 Principle of ItemCF

* If a user likes item $i_1$, then the system recommends other items $i_2$ that are **similar to $i_1$**.

- Estimate users' interest in an unseen candidate item based on the item's similarity to the user-interacted items

    <img src="src/xhs_1.png" width="500"/> 

#### 🔢 Item Similarity sim(item1, item2)

* If there is a large overlap in users who like both $i_1$ and $i_2$, then $i_1$ and $i_2$ are considered **similar**.
* The similarity formula is:

    $
    \text{sim}(i_1, i_2) = \frac{|w_{i_1} \cap w_{i_2}|}{\sqrt{|w_{i_1}| \cdot |w_{i_2}|}}
    $

    where:

    * $w_{i_1}$: set of users who liked item $i_1$
    * $w_{i_2}$: set of users who liked item $i_2$
    * $V = |w_{i_1} \cap w_{i_2}|$: number of users who liked both items

- Embed each item as a sparse vector where each element in the vector represent a user.
- the sim() function is the cosine similarity of the two vectors
- When considering the degree a user likes an item (instead of binary) the formula is below:

    <img src="src/xhs_3.png" width="500"/> 

✅ **Summary:**
ItemCF measures **item-to-item similarity** based on **user co-engagement overlap** (i.e., users who liked both items). The higher the overlap, the more similar the two items are — enabling the system to recommend related content when a user interacts with one.

<img src="src/xhs_2.png" width="400"/>  

Note that itemCF also has variations such as **ContentCF** that derive item similarity based on contents or item features directly instead of using co-engagement data. 

#### Offline Preprocessing

1.  Build **User → Item** Index

    * Record each user’s most recent clicked or interacted item IDs.
    * Given any user ID, you can find the list of items they recently showed interest in.

    <img src="src/xhs_5.png" width="500"/>  

2. Build **Item → Item** Index

    * Compute pairwise similarity between items.
    * For each item, index its top-k most similar items.
    * Given any item ID, you can quickly find its k most similar items.

    <img src="src/xhs_4.png" width="500"/> 

User-item interactions and item-item similarities are computed and stored in indices,
so that during **online serving**, recommendations can be fetched in real time using these precomputed relationships.

#### ⚙️ Online Retrieval (Real-Time Recommendation)

1. Given a **user ID**, use the **User → Item** index to find the list of items the user has recently shown interest in (**last-n** items). i.e. n=200
2. For each item in the **last-n** list, use the **Item → Item** index to find its **top-k similar items**. i.e. k=10
3. For all the retrieved similar items (**up to n×k items**), compute the user’s interest score for each item using a defined formula.
4. Return the **top 100 items with the highest scores** as the final recommendation results.


✅ The precomputed indices from offline processing allows us to quickly find similar items for a user’s recent interactions, estimate interest scores, and output the top-ranked items in real time.

<img src="src/xhs_7.png" width="500"/>  

### 2.2 Swing Model

#### Limitations of ItemCF

If the overlapped users come from a small social circle (like a social media group which often share items), the system could falsely use the signal to predict two items are similar, where in reality they are not.

<img src="src/xhs_8.png" width="500"/>  

#### ⚙️ Core idea of Swing Model

Swing Model adjusts user–item co-occurrence similarity by penalizing users who share too many common items (to reduce overfitting to small groups). It’s an improvement over traditional ItemCF that adds **user-level de-correlation weighting**.

* Let the set of items liked by user $u_1$ be $\mathcal{J}_1$.
* Let the set of items liked by user $u_2$ be $\mathcal{J}_2$.
* Define the **overlap** between two users as:
  
  $$
  overlap(u_1, u_2) = |\mathcal{J}_1 \cap \mathcal{J}_2|
  $$
  
* If users $u_1$ and $u_2$ have a **high overlap**, it means they may come from a **small user circle (tight community)**, so their weight should be **downscaled** to avoid overemphasizing niche clusters.

#### Item Similarity

This is the **core similarity formula of the Swing model**:

* Let the set of users who liked item $i_1$ be $\mathcal{W}_1$.
* Let the set of users who liked item $i_2$ be $\mathcal{W}_2$.
* Define their intersection as $\mathcal{V} = \mathcal{W}_1 \cap \mathcal{W}_2$.
* The similarity between two items is defined as:

  $$
  sim(i_1, i_2) = \sum_{u_1 \in \mathcal{V}} \sum_{u_2 \in \mathcal{V}} \frac{1}{\alpha + overlap(u_1, u_2)}
  $$


✅ **Explanation:**

* It aggregates contributions from all pairs of users $(u_1, u_2)$ who both interacted with the two items.
* The term $overlap(u_1, u_2)$ penalizes users who have high behavioral overlap (i.e., belong to the same tight community).
* $\alpha$ is a smoothing constant to avoid division by zero.
* Intuitively, users who are **too similar** contribute **less** to item similarity — reducing bias toward small, dense user clusters.


#### ⚖️ Comparison Between Swing and ItemCF

* The **only difference** between Swing and ItemCF lies in how they define **item similarity**.
* **ItemCF:**
  If two items share a large proportion of common users, they are considered **similar**.
* **Swing:**
  Additionally considers whether the **overlapping users** come from a **small, dense user circle**.

  * Let $\mathcal{V}$ be the set of users who liked both items.
  * For any two users $u_1$ and $u_2$ in $\mathcal{V}$, define their overlap as $overlap(u_1, u_2)$.
  * If $u_1$ and $u_2$ have a **high overlap**, they are likely from the **same small user circle**, so their **weight is reduced**.

✅ **Summary:**
Swing refines ItemCF by **penalizing correlations caused by tightly connected user groups**, making the similarity measure more robust and less prone to overfitting to niche communities.

### 2.3 UserCF: User Based Collaborative Filtering

#### Principle


* There are many users whose interests are very similar to mine.
* One of these users liked or shared a certain post.
* I haven’t seen this post before.
  
  → **The system recommends this post to me.**

💡 Rednote Example: How does the recommendation system find users whose interests are similar to mine?

* **Method 1:** Users who have a high overlap in the posts they **click, like, favorite, or share**.
* **Method 2:** Users who **follow many of the same authors**.

UserCF finds **like-minded users** based on overlapping behaviors or follow patterns. Then it recommends items (posts, videos, products, etc.) that these similar users have interacted with but I haven’t — leveraging **user-to-user similarity** for discovery.

#### Predicting User Interest

<img src="src/xhs_9.png" width="500"/>  

UserCF assumes *“similar users like similar items.”* It estimates interest in an unseen item by aggregating the preferences of **neighboring users**, weighted by their similarity.

$$\sum_j sim(user, user_j) \times like(user_j, item)$$


#### ⚙️ Calculating Similarity Between Two Users

* Represent each user as a **sparse vector**, where each element corresponds to an item.
* The similarity $sim$ between two users is the **cosine of the angle** between their vectors.



**UserCF** measures **user–user similarity** using **cosine similarity** over their interaction sets — i.e., the more overlapping items two users like (normalized by how many items each has liked), the more similar their interests are.

* Let the set of items liked by user $u_1$ be $\mathcal{J}_1$.
* Let the set of items liked by user $u_2$ be $\mathcal{J}_2$.
* Define their intersection as $I = \mathcal{J}_1 \cap \mathcal{J}_2$.
* The similarity between two users is defined as:

  $$
  sim(u_1, u_2) = \frac{|I|}{\sqrt{|\mathcal{J}_1| \cdot |\mathcal{J}_2|}}
  $$

#### Reducing the Weight of Popular Items

$$
sim(u_1, u_2) = \frac{\sum_{l \in I} \frac{1}{\log(1 + n_l)}}{\sqrt{|\mathcal{J}_1| \cdot |\mathcal{J}_2|}}
$$

where:
- $n_l = \text{the number of users who liked item } l$, which reflects the **popularity level of item $l$**.

- Items that are liked by many users (high $n_l$) contribute **less** to user similarity. 
- Rare or niche items (low $n_l$) contribute **more**, helping the system discover more *personalized* relationships instead of those dominated by universally popular content.

Here’s the extracted and translated content from the image:


#### 🧮 Offline Preprocessing

1. Build a **User → Item** Index (same as ItemCF)

    * Record each user’s most recent clicked or interacted item IDs.
    * Given any user ID, you can find the list of items they have recently shown interest in.

2. Build a **User → User** Index

    * For each user, index their top-k most similar users.
    * Given any user ID, you can quickly find their k most similar users.

    <img src="src/xhs_10.png" width="500"/>   

user–item and user–user relationships are computed and stored, so that during **online serving**, the system can quickly look up similar users and recommend items they liked.


#### ⚙️ Online Retrieval (Real-Time Recommendation)

1. Given a **user ID**, use the **User → User** index to find the **top-k similar users**.
2. For each of these top-k similar users, use the **User → Item** index to find the list of items they have recently shown interest in (**last-n items**).
3. For all the retrieved **n×k similar items**, estimate the target user’s **interest score** for each item using a defined formula.
4. Return the **top 100 items** with the highest scores as the recommendation results.

    <img src="src/xhs_11.png" width="500"/>  

### 2.4 Two-Tower Model: Architecture and Training

#### Concept

The **Two-Tower Model** (also known as **Dual Encoder** or **Siamese Architecture**) is a large-scale **retrieval model** used in recommendation and search systems.
It learns **vector representations (embeddings)** for both users and items, enabling efficient **similarity-based retrieval** via inner product or cosine similarity.

#### 1. User Tower

* **Inputs:**

  * User ID
  * Discrete user features (e.g., gender, location, device)
  * Continuous user features (e.g., activity level, time since last login)
* **Processing:**

  * Embedding layers for categorical features
  * Normalization / bucketing for continuous features
  * Concatenation of features → passed through a neural network

* **Output:** User embedding vector **a**

    <img src="src/xhs_12.png" width="500"/>   

#### 2. Item Tower

* **Inputs:**

  * Item ID
  * Discrete item features (e.g., category, brand)
  * Continuous item features (e.g., popularity, price)
* **Processing:**

  * Embedding + transformation + neural network
* **Output:** Item embedding vector **b**

    <img src="src/xhs_13.png" width="500"/>  

#### 3. Similarity Computation

<img src="src/xhs_14.png" width="500"/> 

$$\text{score} = a \cdot b \quad \text{or} \quad \cos(a, b) = \frac{a \cdot b}{|a||b|}$$

This score represents the **predicted interest** or **relevance** between the user and item. Cosine is more commonly used now.

#### Training Methods

(1) **Pointwise Training**

* Treats retrieval as a **binary classification** task.
* Each (user, item) pair is labeled as **positive** (clicked) or **negative** (not clicked).
* Encourages:

  * Positive samples → cosine similarity ≈ +1
  * Negative samples → cosine similarity ≈ −1
* Typical positive:negative ratio = 1:2 or 1:3 (industry convention)

(2) **Pairwise Training**: Sample one positive and negative item pair for a user

<img src="src/xhs_15.png" width="500"/>  

* Shared parameters across positive and negative item encoders.
* Objective: Encourage cosine similarity with positive item > negative item by a margin $m$. That is, make sure the predicted user interest in positive item is higher than that in negative item.
* Uses **Triplet-based** loss with one user, one positive item, one negative item

    If $\cos(a, b^+)) > \cos(a, b^-))+m$ &rarr; no loss; 
    
    Else, loss is measured below:

    - **Triplet hinge loss:**

    $$L(a, b^+, b^-) = \max(0, \cos(a, b^-) + m - \cos(a, b^+))$$


    - **Triplet logistic loss:**

    $$L(a, b^+, b^-) = \log(1 + \exp[\sigma(\cos(a, b^-) - \cos(a, b^+))])$$

(3) **Listwise Training**

* One user → one positive sample + multiple negative samples.
* Encourages:

  * $\cos(a, b^+)$ → large
  * $\cos(a, b^-)$ → small

**Procedure:**

1. Compute cosine similarities between user embedding and all (positive + negative) items.
2. Apply **Softmax** over similarities to get probabilities $s_i$.
3. Use **Cross-Entropy Loss**:
    $$L = -\log(s_{positive})$$
   
    <img src="src/xhs_16.png" width="500"/>  

#### Negative Sampling Strategies

* **Positive samples:** Items the user actually clicked.
* **Negative samples:** Several possible definitions:

  1. Items not retrieved at all.
  2. Items retrieved but filtered out by ranking.
  3. Items shown but not clicked (exposed negatives).

References:

* Jui-Ting Huang et al., *Embedding-based Retrieval in Facebook Search*, KDD 2020.
* Xinyang Yi et al., *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations*, RecSys 2019.



#### Comparison: Two-Tower for Retrieval vs Ranking 

Below model architecture is designed for ranking and not for retrieval:

<img src="src/xhs_17.png" width="500"/>  

| Aspect      | Two-Tower for Retrieval                    | Ranking  |
| ----------- | ---------------------------------- | --------------------------- |
| Structure   | Two separate encoders (user, item) | Single concatenated network |
| Efficiency  | Enables ANN search (Faiss, ScaNN)  | Requires full pair input    |
| Application | Retrieval / Recall stage           | Ranking / Scoring stage     |
| Output      | Separate embeddings                | Combined interest score     |

---

## 🧠 Summary

* The **two-tower model** outputs:

  * **User embedding** (from user tower)
  * **Item embedding** (from item tower)
* Their **cosine similarity** predicts user interest.
* Supports **three training paradigms:**

  1. **Pointwise:** single pair (binary classification)
  2. **Pairwise:** one positive + one negative (triplet loss)
  3. **Listwise:** one positive + multiple negatives (softmax CE loss)
* Scalable to large corpora — efficient for **ANN-based retrieval**.





