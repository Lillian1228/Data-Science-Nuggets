## 1. Binary Recommendation Metrics

Binary recommender metrics directly measure the click interaction.

### 1.1 CTR

Reward-focused metrics at item-impression level - Direct estimation (“matching”) measures the accuracy of the recommendations over the subset of user-item pairs that appear in both actual ratings and recommendations.

### 1.2 AUC (Area under the ROC Curve)

AUC is the probability that the recommender will rank a randomly chosen relevant/clicked instance higher than a randomly chosen non-relevant/not-clicked one over the subset of user-item pairs that appear in both actual ratings and recommendations.

- Inputs

    * **Predicted scores** for each item (e.g., probability that a user clicks).
    * **Ground-truth labels** (1 if the item was clicked/relevant, 0 if not).

- Steps

    1. **Pair all positive and negative items** (relevant vs non-relevant).
    2. **Count “correctly ranked pairs”**: a positive item is ranked higher than a negative item.
    3. **Compute AUC**:

    $$
    \text{AUC} = \frac{\text{\# correctly ranked positive-negative pairs}}{\text{total \# of positive-negative pairs}}
    $$

* Result ranges **0–1**:

  * 0.5 → random ranking
  * 1.0 → perfect ranking
  * <0.5 → worse than random

* Can also compute using **ROC curve**:

  * Plot True Positive Rate (TPR) vs False Positive Rate (FPR) at multiple thresholds.
  * AUC = area under this curve.
* Works with **binary relevance**, **Insensitive to top-K cutoff**: it measures overall ranking quality, not just the top of the list. Offline AUC is often weak proxy.

### 1.3 🟢 Hit Rate @K (HR\@K)

**What it measures:**

* Fraction of users for whom **at least one relevant item** appears in top-K recommendations:

$$
\text{HR@K} = \frac{\#\{\text{users with ≥1 relevant item in top K}\}}{|U|}
$$
- Similar to CTR but at the user level, and does not care how many items the user clicked.

**When to use:**

* Binary indicator for **“does the recommendation list contain something the user likes?”**
* Useful in practical recommender A/B tests.

**Limitations:**

* Doesn’t account for **number of relevant items** or **ranking order**.
* Can saturate quickly if top-K is large relative to number of relevant items.

## 2. Ranking Metrics

Ranking metrics reward putting the clicked items on a higher position/rank than the others. 

### 2.1 Precision\@K



**What it measures:**
- how consistently a model is able to pinpoint the items a user would interact with. 
* Fraction of **top-K** recommended items that are relevant.

$$
\text{Precision@K} = \frac{\#\{\text{relevant items in top K}\}}{K}
$$

**When to use:**

* If you care about **relevance density** in the top of the list (user’s immediate attention span).
* Good for search, top-banner recommendations.

**Limitations:**

* Ignores items beyond K.
* Doesn’t care whether you missed relevant items, only how many you hit in top-K.

### 2.2 🟢 Recall\@K

**What it measures:**
- Whether a model can capture all the items the user has interacted with. 
- Sensitive to K: If K is large enough, a recommender system can get a high recall even if it has a large amount of irrelevant items, if it has also identified the relevant items.
* Fraction of **all relevant** items that appear in the **top-K list**.

$$
\text{Recall@K} = \frac{\#\{\text{relevant items in top K}\}}{\#\{\text{all relevant items}\}}
$$

**When to use:**

* If you want to measure **coverage of user interests**.
* Useful when users might engage with multiple items (e.g., a product catalog, content feed).

**Limitations:**

* Can be inflated by recommending long lists.
* Doesn’t reward ranking order — as long as relevant items appear in the list, recall increases.
- Offline evaluation uses **future interactions** from the test set as a proxy for relevance. If the test set has **more relevant items than K** for a user, Recall@K is **capped** below 1.0. 
- In practice, comparing models' recalls on the same test set is still meaningful. Report multiple Ks (i.e. 5, 10) to capture different perspectives. 

### 2.3 MAP (Mean Average Precision)

**What it measures:**
- position-sensitive version of Precision@k, where getting the top-most recommendations more precise has a more important effect than getting the last recommendations correct. 
- When k=1, Precision@k and MAP@k are the same.
* Average of **precision at each relevant item position** per user, then averaged across users:

    $MAP@k = \frac{1}{\left | A \right |} \sum_{i=1}^{\left | A \right |} \frac{1}{min(k,\left | A_i \right |))}\sum_{n=1}^k Precision_i(n)\times rel(P_{i,n})$

    - $rel(P_{i,n})$ is an indicator function that produces 1 if the predicted item at position n for user i is in the user’s relevant set of items.
    - Average Precision (AP) for the user = average of these precision values among the maximum possible number of relevant items: ${min(k,\left | A_i \right |))}$


**When to use:**

* Evaluates both **ranking quality and coverage** of all relevant items.
* Good for multi-item recommendation scenarios.

**Limitations:**

* Sensitive to incomplete relevance labels — if some relevant items are unknown, MAP is underestimated.
* More complex to interpret than Precision\@K or Recall\@K.

### 2.3 🟡 NDCG (Normalized Discounted Cumulative Gain)

**What it measures:**
- Relevance of the ranked recommendations **discounted by the rank** at which they appear. It is normalized to be between 0 and 1. 
- Improving the highest-ranked recommendations has a more important effect than improving the lowest-ranked recommendations.
* **Ranking quality** by giving higher weight to relevant items at higher ranks.

    $$
    \text{DCG@K} = \sum_{r=1}^{K} \frac{rel(P_{i,r})}{\log_2(r+1)}
    $$

    $$
    \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
    $$

    - where IDCG = ideal DCG, perfect ranking: $$\text{IDCG@K} = \sum_{r=1}^{K} \frac{1}{log_2(r+1)}$$

**When to use:**

* Best metric when **order matters** (e.g., top slot more valuable than 5th slot).
* Standard for ranking-based recommenders, search results, content feeds.

**Limitations:**

* Requires graded relevance if possible; if only binary labels exist (clicked vs. not), it reduces to simpler form.
* Slightly less interpretable for business stakeholders than precision/recall.

### 2.4 🔵 MRR (Mean Reciprocal Rank)

**What it measures:**

* Average of the reciprocal rank of the **first relevant item** in each user’s recommendation list:

$$
\text{RR}_u = \frac{1}{\text{rank of first relevant item for user } u}, \quad \text{MRR} = \frac{1}{|U|}\sum_u \text{RR}_u
$$

**When to use:**

* When you care about **how quickly a user finds a relevant item**.
* Top-1 / top-priority recommendation scenarios.

**Limitations:**

* Ignores all other relevant items after the first.
* Sensitive to a single highly ranked hit; may overestimate usefulness if multiple items matter.

## 3. Diversity Metrics

### 3.1 🔴 Inter-List Diversity

**What it measures:**

$$
\text{ILD@K} = \frac{\sum_{i,j, \{u_i, u_j\} \in I}(cosine\_distance(R_{u_i}, R_{u_j}))}{|I|}
$$


where: 
- $u_i, u_j \isin U$ denote the i-th and j-th user in the user set. 
- $R_{u_i}$ is the binary indicator vector representing provided recommendations for $u_i$
- $I$ is the set of all unique user pairs.
- Inter-List Diversity@k measures the inter-list diversity of the recommendations when only k recommendations are made to the user.

**When to use:**

- It measures how user’s lists of recommendations are different from each other. This metric has a range in [0, 1]
- The higher this metric is, the more diversified lists of items are recommended to different users.


**Limitations:**

* Requires a well-defined similarity/distance metric.
* Too much diversity may reduce relevance — must balance against precision.

### 3.2 🟡 Intra-List Diversity (ILD or ILD\@K)

**What it measures:**
- Given a list of items recommended to one user and the item features, the averaged pairwise cosine distances of items is calculated. Then the results from all users are averaged as the metric Intra-List Diversity@k. 
* Similar to inter-list diversity but per-item:

    $$
    \text{ILD@K} =\frac{1}{N}\sum_{i=1}^N \frac{\sum_{p, q, \{p, q\} \in I^{u_i}}(cosine\_distance(v_p^{u_i}, v_q^{u_i}))}{|I^{u_i}|}
    $$
    where:
    - $u_i \isin U$ denote the i-th user in the user set of size $N$.
    - $v_p^{u_i}, v_q^{u_i}$ are the item features of the p-th and q-th item in the list of items recommended to $u_i$, $p, q \isin {0,1,...,k-1}$
    - $I^{u_i}$ is the set of all unique pairs of item indices for $u_i$
- Range in [0, 1]. The higher this metric is, the more diversified items are recommended to each user.

**When to use:**

* Ensures recommendations are **not too redundant**, increasing exploration and user satisfaction.

**Limitations:**

* Needs a good distance/similarity measure between items.
* High diversity can conflict with relevance (too exploratory).

## 4. ✅ Practical Tips

### Choosing K

1. Match the actual UI experience
    - Should K be more than the number of items actually shown?

        Sometimes yes, during offline evaluation:
        - You may generate and score say top 100 candidates, even if the UI only shows 10. This helps compare different models more fairly before truncating to UI size.

2. Use multiple K values for robustness
3. Business goals: smaller K is critical if the goal is quick click-thru; larger K matter more if the goal is catalog exploration.


### Choosing Metrics

* **Precision\@K** → relevance density at the top.
* **Recall\@K** → breadth of relevance coverage.
* **NDCG** → ranking quality (order sensitivity).
* **ILD** → diversity, avoids redundancy.

A balanced evaluation usually reports **NDCG\@K + Precision\@K** (effectiveness) and **ILD** (diversity), sometimes also **Recall\@K** when multi-relevant-item coverage matters.

In sparse CTR scenario, metrics that aggregate per user (Hit@K, MAP, MRR, Hit Rate) are generally more robust than per-impression CTR.

---

| Metric                          | Category            | Conceptual Computation                                      | When to Use                                                                          | Limitations                                                          | Suitable for                  | Robust for Sparse CTR? |
| ------------------------------- | ------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------- | ----------------------------- | ---------------------- |
| CTR                             | Engagement          | Fraction of clicked impressions                             | Track immediate engagement                                                           | Sparse clicks → high variance                                        | Online                        | No                     |
| Conversion Rate / Purchase Rate | Engagement          | Fraction of users completing downstream actions             | Track business outcomes                                                              | Rare events may need large sample                                    | Online                        | Yes                    |
| Session Engagement / Dwell Time | Engagement          | Avg time spent on recommended items                         | Track engagement quality                                                             | Sensitive to outliers                                                | Online                        | Yes                    |
| User Retention / Repeat Visits  | Engagement          | Fraction of users returning after exposure                  | Long-term engagement                                                                 | Requires time lag; external influence                                | Online                        | Yes                    |
| Hit\@K                          | Ranking / Relevance | 1 if user sees ≥1 relevant item in top-K                    | Top-K relevance evaluation                                                           | Ignores multiple relevant items                                      | Offline                       | Yes                    |
| Precision\@K                    | Ranking / Relevance | Fraction of top-K items relevant                            | Accuracy at top of list                                                              | Sensitive to K; ignores missed items                                 | Offline                       | Yes                    |
| Recall\@K                       | Ranking / Relevance | Fraction of relevant items in top-K                         | Coverage of relevant items                                                           | Sparse labels reduce reliability                                     | Offline                       | Partial                |
| NDCG\@K                         | Ranking / Relevance | Discounted gain by position of relevant items               | Rank-aware evaluation                                                                | Sensitive to label quality                                           | Offline                       | Partial                |
| MRR                             | Ranking / Relevance | Reciprocal rank of first relevant item                      | First-hit relevance                                                                  | Ignores later relevant items                                         | Offline                       | Yes                    |
| MAP                             | Ranking / Relevance | Mean of per-user average precision                          | Overall ranking quality                                                              | Incomplete labels reduce accuracy                                    | Offline                       | Yes                    |
| Intra-list Diversity            | Diversity           | Avg dissimilarity between items in a single user’s list     | Variety/serendipity within a user’s recommendations                                  | Hard to balance with accuracy                                        | Offline                       | N/A                    |
| **Inter-list Diversity (ILD)**  | **Diversity**       | Avg dissimilarity between recommendation lists across users | Use when you want **different users** to see different content, avoid homogenization | Computationally heavy; may conflict with fairness or popularity bias | Offline (periodic monitoring) | N/A                    |
| Hit Rate                        | Engagement          | Fraction of users with ≥1 relevant recommendation           | Simple user-level success                                                            | Binary; ignores ranking                                              | Offline                       | Yes                    |
| AUC                             | Ranking / Relevance | Probability random positive ranks above random negative     | Overall ranking quality                                                              | Insensitive to top-K; assumes binary labels                          | Offline                       | Partial                |

---



