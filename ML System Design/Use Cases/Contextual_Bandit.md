
### Stage 2: Use CMAB to make content recommendations directly

### 1. Business Problem

During the 90-day onboarding window for newly enrolled 401(k) participants, the homepage has limited space and too many potential messages. We want to cut noise and surface the most helpful content so users complete key 401(k) tasks and also discover relevant benefits beyond 401(k).

#### Business Objective
The business success metric is focusing on the increase in engagement (CTR) among customers who have seen a personalized recommendation vs. randomized baseline. Given a limited and stable number of contents, can we boost the engagement over randomized policy baseline.

- Primary objective (optimization target): 
  
  Maximize cumulative engagement value per onboarder over 90 days:

    - near-term: clicks on messages
    - downstream: task completions (e.g., set contribution rate, pick investments, name beneficiaries)

- Secondary goals (guardrails):
    - Healthy diversity/coverage of “beyond 401(k)” topics
    - User experience: limit repetition/fatigue
  
#### Key constraints
  - Eligibility/compliance: only show content a user is eligible for
  - Frequency caps per message per user (e.g., ≤4 exposures in a month)

    &rarr; Eligibility and frequency capping need not to be handled by the MAB model at the message ranking stage as they happen during the re-ranking stage controlled by rec engine. But they could affect what the users see and their effects are baked into the system inputs.  

  - Single slot on homepage for 30 NBA messages

  - Limited customer data during initial onboarding period i.e. missing income, assets, age etc. that can be used for personalization

### 2. Choosing the right method: Why MAB?

- Time sensitivity to exploit and maximize the rewards i.e. recommending holiday deals
- New contents with little historical performance data need to be explored
- Personalization under limited data and signals. 
  - Traditional ML methods are not appropriate as they require a wide range of data points to learn individual customers' preferences/interests. New participants often lack such complex signals. 
  - Contextual bandits can leverage limited data to create neighborhoods,  learn the segmentwise differences, and suggest different best actions based on the neighborhood.

- Greedy nature and limitation on take long-term effects of actions into account
  
  Policies derived from the contextual bandit formulation are greedy in the sense that they do not take long-term effects of actions into account. These policies effectively **treat each visit to a website as if it were made by a new visitor uniformly sampled from the population of the website’s visitors**. By **not using the fact that many users repeatedly visit the same websites**, greedy policies do not take advantage of possibilities provided by long-term interactions with individual users.

    &rarr; Not a concern in our use case since the policy only runs during the initial 90 days for each new participant.

    
### 3. Frame the Problem as ML Task

- ML Objective

    Maximize the rewards (reward is 1 if click and 0 otherwise) by showing the best performing message selected by CMAB.

- System Inputs
    - Context (x). Snapshot at decision time
        - User state: 
          - demographics
          - days since enrollment, geo/time-of-day, device, language
          - prior tasks done, prior clicks, prior impressions
        - Plan-level: employer features (industry, size, enrollment type)
    - Arms (A): Eligible messages for this user at this moment - each with metadata: 
      - topic, format, risk category, historical CTR, novelty age, campaign priority
    - Rewards (r): A single scalar per impression or per session factoring multiple objectives (clicks and task completions).
      - Weights are tunable; use business value or Shapley-style attribution. Time-decay rewards so earlier completions are worth more.
      -  Assign delayed rewards back to their triggering message with a reasonable window (e.g., 7–14 days) and last-touch or fractional credit.

- System: Policy π(a|x). A contextual slate bandit choosing K items.
  - Base: Linear/Logistic UCB or Thompson Sampling (contextual) for per-message click/action probability
  - Slate: position-aware scoring (cascade/Plackett-Luce) + diversity bonus and frequency caps
  - Exploration: enforced ε-greedy floor or posterior sampling; per-arm minimum exposure

- System Outputs


    - Action. Select a slate of K messages and an order (position matters) for each participant.

### 4. Data Preparation

1.	Event data. Log each decision with: user_id (hashed), timestamp, context features, chosen action, slot position.
2.	Outcome data. Join impressions to outcomes (clicks; task events) with windows & rules; compute reward r and time-to-event.
3.	Features.
    - User/session: days-since-enroll buckets, prior exposures/clicks (short & long windows), device, time buckets.
    - Arm/message: topic, format, novelty age, historical uplift, “beyond-401(k)” flag.
    - Interactions: (days-since-enroll × topic), (device × format).

#### Temporal Split

We gathered recent 5 months' clickstream data and split it into `train / val / test` based on recency. Initial 3 months was reserved for training, 4th month for validation, and last month for holdout test. 


The reward signal in this domain is very sparse because users usually do not click on an engager, and user clicking is very random so that returns have high variance.



### 5. Offline Policy Evaluation


#### 5.1 Replay + Global/Neighbor Average Imputation

- Problem: Historical data (actions and rewards) were generated by a different policy. We need some way to estimate the expected reward of the new policy based on history.

- **Idea:**

    * You "replay" the logged dataset through the target policy.
    * Whenever the action chosen by the target policy matches the action actually taken in the logged data, you use the observed reward; 
    - When the predicted arm is different, the mean reward from the same neighborhood in the training data is used.

- Why replay Works in this case

    - a substantially exploratory behavior policy (≈20% of traffic was random) &rarr; reduce bias

    - high overlap between behavior and candidate policies

    - large dataset (>100k impressions) with 30 actions &rarr; ensure good sample size and limit variance

- Why Other Unbiased Estimator Methods are not favored?

    DM, IS, and DR require building a reward model or knowing the propensity from the behavior policy (modeling the propensity if not available), which are not practical or worth the time efforts in this use case.

    - Reward model = **low ROI**: weak accuracy under sparse data + not reused online.
    * Behavior policy model = **low necessity**: one-time effort for offline simulation, have to model it given no logging propensity, exploratory logging (20% random) gives coverage - replay along is feasible.
    * Replay + neighbor-average imputation = **simpler, more robust fit** for your setting, with A/B testing as the ultimate truth.
    

#### Quick notation / assumptions


* `embed(x)` = compact context embedding used for both linear models and neighbor search (PCA or small supervised NN).
* `K` (neighbors) default = **100**, `min_neighbors` = **30**.
* Imputation: weighted neighbor mean of historical rewards for (x,a).
* Use **replay + impute** for all offline evaluations: if `pi(x)==a_logged` use `r`; else use neighbor-imputed reward or global action mean fallback.

#### 5.2 Policy Training (on TRAIN)

The initial benchmarking of various policies was all done on just the training data using 5-fold cross validation. This stage determined the best model for our task. 

All policies use the same context embedding `z = embed(x)`.

A. **Fit a linear reward model Per Arm**

* Fit ridge logistic (or ridge regression on CTR) per action. Regularization `λ` default = **1.0**. 
* This model is used to produce point estimates $\hat\mu_a(z)$ and to initialize LinUCB / linTS.

B. **LinGreedy (greedy linear)**

* Policy: pick $a = \arg\max_a \hat\mu_a(z)$.
* No exploration parameter.

C. **LinUCB**

* Maintain $A_a = \lambda I + \sum_{i: a_i=a} z_i z_i^\top$ and $b_a = \sum_{i: a_i=a} z_i r_i$ from train.
* Estimate $\hat\theta_a = A_a^{-1} b_a$. For context z choose $a = \arg\max_a (z^\top \hat\theta_a + \alpha \sqrt{z^\top A_a^{-1} z})$.


D. **linTS (linear Thompson Sampling)**

* Use Bayesian linear regression per action (Gaussian prior with variance `σ0^2` and noise variance `σ^2`). Initialize with training sufficient stats A\_a, b\_a.
* At evaluation time sample $\tilde\theta_a \sim \mathcal{N}(A_a^{-1}b_a, \, \sigma^2 A_a^{-1})$ (or use prior scale). Pick $a = \arg\max_a z^\top \tilde\theta_a$.
* Tune prior/noise scale `σ` grid: **{0.1, 1, 2}**; use 20 samples per context for stable TS evaluation or 1 sample for cheap approximate TS.

E. **Neighborhood policies (UCB / ε-greedy / TS variants using neighborhood stats)**

* For each (z,a) compute neighborhood stats among training rows: weighted mean $\bar r_{K}(z,a)$ and empirical variance $s^2_{K}(z,a)$.
* Neighborhood **Greedy**: pick argmax $\bar r_K(z,a)$.
* Neighborhood **ε-greedy**: with prob ε pick random action, else greedy. Tune ε ∈ {0.01, 0.05, 0.1}.
* Neighborhood **UCB**: pick $a = \arg\max_a [\bar r_K(z,a) + \beta \cdot s_K(z,a)/\sqrt{n_{K}(z,a)}]$. Tune `β` ∈ {0.5, 1, 2}.
* Neighborhood **TS**: treat Beta/Bernoulli posterior per (z,a) approximated from neighbors (α = successes+1, β = failures+1); sample CTR and pick max. (Works well with sparse CTR if K moderate.)



#### 5.3 Hyperparameter tuning (on VAL)

Once the model was chosen we then performed hyperparameter tuning on the validation set and select the best tune model based on objectives below.

* **Random grid search** over:

  * LinUCB `α`: [0.1, 100]
  * linTS `σ`: {0.1,1,2}
  * Ridge `λ`: [1000, 5000000]
  * Neighborhood `K`: {30,50,100,1000}
  * Neighborhood `min_neighbors`: {30,50,100}
  * ε: {0.01,0.05,0.1}; β (UCB) {0.5,1,2}

* **Tuning objective on VAL**: Estimated policy rewards (CTR)

    - Use replay+neighbor-avg to impute `mean_reward_val`. 
    - Compute the `std_error_val` using the sample variance of rewards in the validation set.
    - Score = `mean_reward_val - λ_risk * std_error_val` with `λ_risk = 1` (risk-averse).

- Report support diagnostics: `fraction_replayed`, `fraction_neighbors_imputed`, `low_confidence_fraction` 
    - Support = how much your logged dataset “covers” the actions that a candidate policy would choose
    - High-confidence: if n_neighbors ≥ min_neighbors (i.e. 100) and CI width < 0.005 CTR (or <20% relative to mean if you prefer a relative rule).

    - Medium-confidence if neighbors ≥ min_neighbors but CI wider.

    - Low-confidence if neighbors < min_neighbors (fallback to global action mean).

- Other Recommender Metrics: Hit Rate, Precision@K, Recall@K, NCDG@K, Inter-list Diversity

- **Ranking:** rank policies by `mean_reward_pi` and prefer those with smaller `std_error` and lower `low_confidence_fraction`.

* Pick top N (e.g., N=3) parameterized policies for final test evaluation.

Actual Results:
- Low support for LinUCB and Thompson Sampling
- No model outperformed LinGreedy so moved with it into hyperparameter tuning


#### 5.4 Offline evaluation (on TEST)

For final candidate policy `π`:

1. For each test row i estimate the reward:

   * compute action `a = π(z_i)` (use deterministic choice for greedy/UCB; for TS sample seed-fixed draws for repeatability).
   * if `a == a_logged_i`: `reward_used = r_i` (replay).
   * else: `reward_used = neighbor_impute(z_i, a, K)` (weighted mean among K neighbors who saw `a`), or global action mean if neighbors < `min_neighbors`. Flag low-confidence rows.
2. Compute `mean_reward_pi = mean(reward_used)` and bootstrap 95% CI (1000 resamples).
    - User-level bootstrapping instead of impression-level: prevents within-user correlation


        For $b = 1 \dots 1000$:

        1. Sample a dataset $D_b$ by drawing $n$ **users (clusters)** *with replacement* from $D$.
        2. Recompute $\hat{R}(D_b, \pi)$ using replay+impute on $D_b$.
        3. Collect $\hat{R}_b$.

    - Construct confidence interval

        * Sort $\{\hat{R}_1, \dots, \hat{R}_B\}$.
        * Take the 2.5th and 97.5th percentiles → 95% bootstrap CI.
        * Report mean estimate = $\hat{R}(D,\pi)$, CI = \[p2.5, p97.5].

3. Report on Recommender Metrics: Hit Rate, Precision@K, Recall@K, NCDG@K, Inter-list Diversity


#### 5.5 Multi-objective method comparison

On our best model, we also tested several methods below to create the multi-objective rewards:

- Baseline Single-Objective Model
    - Train a baseline, CTR-only MAB model with production configuration.
- OR Model
    - Train a MAB model on logical OR label aggregation: CTR OR task_completed
- Linear Label Aggregation
    - Train a MAB model on linearly aggregated labels:
    - Label: α * Click +(1-α) task_completed where α is a hyperparameter in [0,1] interval.


We trained one model targeting baseline CTR label & OR labels, respectively.
We then trained 50 models for each alpha on a [0,1] interval, and averaged resulting metrics on 0.1 interval (5 models per interval).

We selected the best models for each method above based on the validation set. Then we evaluated performance metrics on test data such as CTR@1, Task_Completion@1, Precision@1 for CTR, and Precision@1 for Task_Completion.

Actual Results:
- Linear label aggregation with alpha between 0.6-0.7 (best tuned from validation) performed the best considering both CTR@1 and Task_Completion@1 
- CTR is almost equal to CTR-only baseline, whereas task completion rate is 26% higher.


### 6. Deployment and Serving

#### LinGreedy Recap

* It’s basically **LinUCB without the confidence bonus**.
* The model is a **linear regression per arm** (or pooled with shared weights), learning:

$$
\hat r(x,a) = \theta_a^\top x
$$

* At inference, you **score each eligible arm** with its linear model, pick the arm(s) with the highest predicted reward.
* To ensure exploration, add an **ε-greedy rule**: with probability ε, pick a random eligible arm; with probability $1-ε$, pick greedy best.


#### 6.1 Daily Batch Scoring and Model Update

1. Collect logs from previous day
    * Contexts shown $(x_i)$
    * Arm chosen $(a_i)$
    * Observed reward $(r_i)$ (click, task completion, etc.)
    * Join with user + message features.
    * Filter for eligible onboarding users (within 90 days).

2. Update linear models Incrementally 

    - Warm-start from offline model initially: We kept $\theta$ from the offline dataset as the baseline.
    - Weights update as new daily data coming in using **Recursive Least Square**

    **Arm-specific RLS**
    - For each arm $a$, maintain its own coefficient vector $\theta_a$ and inverse covariance matrix $P_a$ (tracks uncertainty in estimates).
    - Run RLS updates sequentially for each new sample in yesterday’s data
    - We also applied a forgetting factor of 0.99 to decay the weight of older impressions.

    Advantages of RLS:

    * ✅ **Faster than full Retrain**: equivalent to “online learning"
    * ✅ **Stable**: starts from large offline model, adjusts gently.
    * ✅ **Adaptive**: responds to novelty (new arms, behavior shifts).
    * ✅ **Granular**: per-arm RLS lets each message learn at its own pace.

3. Generate candidate slates (batch inference)

    For each active user:

    * Build context vector $x_u$ (days since enrollment, device, prior actions).
    * For each eligible message $a$: compute $\hat r(x_u,a)$.
    * With probability ε: randomly shuffle eligible arms.
    * Otherwise: pick top-K arms by $\hat r(x_u,a)$.
    * Apply re-ranking rules (eligibility, caps, diversity, campaign priorities).

4. Write outputs to serving table

    * Store per-user slate in a **recommendation table** (e.g., key = `user_id × device`).
    * This is what the homepage/app tiles read the next day.


#### 6.2 Online Evaluation Process

Since policy updates **once per day**, online evaluation must respect that cadence.

1. Define experiment design

    * **Treatment group** → batch LinGreedy policy.
    * **Control group** → baseline (e.g., random policy or business-as-usual heuristic).
    * Randomize assignment at **user level** (not impression level) to avoid contamination.

2. Logging (critical for replay & guardrails)

    For each impression, log:

    * User context snapshot.
    * Candidate set of eligible messages.
    * Chosen slate + slot positions.
    * **Probability of chosen slate under policy** $\pi(a|x)$
        - in LinGreedy: either 1-ε for greedy choice, ε/|A| for random arms).
    * Observed reward events (clicks, task completions, etc.).

3. Online metrics to monitor

    * **Primary (optimization target):**

        * CTR uplift vs. control.
        * Task completion rate uplift.
    
    * **Secondary (guardrails):**

        * Diversity across “beyond 401(k)” topics.
        * User fatigue (repetition rate, dwell time).
    * **Operational health:**
        * % of users with non-empty slates.
        * ε fraction actually randomized.
        * Arm coverage (all arms explored daily).

4. Evaluation cadence

    * Compute **daily aggregates** (CTR, task completion, reward per impression).
    * Use sequential testing / CUPED adjustment to get faster read.
    * Because updates are daily, monitor **trend of uplift across days**, not per-impression.



### 7. Limitations and Future Directions 

1. Greedy optimization: maximizing only the immediate clicks and did not take the long-term effects of recommendations into account.

    