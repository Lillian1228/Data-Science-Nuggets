
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
  - Slots K for different devices: web - 3 tiles at different positions, app - 2 tiles.

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
    - Rewards (r): A single scalar per impression or per session with delayed credit.
      -  r = 0.05 * click + 1.0 * key_task_completed + 0.2 * qualified_view_benefit
      - Weights are tunable; use business value or Shapley-style attribution. Time-decay rewards so earlier completions are worth more.
      -  Assign delayed rewards back to their triggering message with a reasonable window (e.g., 7–14 days) and last-touch or fractional credit.

- System: Policy π(a|x). A contextual slate bandit choosing K items.
  - Base: Linear/Logistic UCB or Thompson Sampling (contextual) for per-message click/action probability
  - Slate: position-aware scoring (cascade/Plackett-Luce) + diversity bonus and frequency caps
  - Exploration: enforced ε-greedy floor or posterior sampling; per-arm minimum exposure

- System Outputs


    - Action. Select a slate of K messages and an order (position matters) for each participant.

### 4. Data Preparation

1.	Event data. Log each decision with: user_id (hashed), timestamp, context features, chosen slate & order, per-item propensities, slot positions.
2.	Outcome data. Join impressions to outcomes (clicks; task events) with windows & rules; compute reward r and time-to-event.
3.	Features.
    - User/session: days-since-enroll buckets, prior exposures/clicks (short & long windows), device, time buckets.
    - Arm/message: topic, format, novelty age, historical uplift, “beyond-401(k)” flag.
    - Interactions: (days-since-enroll × topic), (device × format).

- Data Challenge

    The reward signal in this domain is very sparse because users usually do not click on an engager, and user clicking is very random so that returns have high variance.

why per-item propensities?
how to handle rewards at different positions: would randomization solve?
batch scoring instead of real-time



### 5. Model Development


- Policy Selection based on Offline Simulation




- Methods

    1. Greedy optimization: maximizing only the immediate clicks and did not take the long-term effects of recommendations into account.

        Use a supervised model to estimate the probability of a click as a function of user features. Then we define an "epsilon-greedy policy that selected with probability 1-epsilon the engager predicted by the supervised model to have the highest probability of producing a click, and otherwise selected from the other engagers uniformly at random.



### 6. Deployment and Serving

- Batch Scoring

