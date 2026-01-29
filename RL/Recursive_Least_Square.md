

### What is Recursive Least Squares (RLS)?

RLS is an **online closed form version of linear regression**.

* In ordinary least squares, you solve once with normal equation:

$$
\theta = (X^\top X + \lambda I)^{-1} X^\top y
$$

* But recomputing this from scratch every day with all history is expensive.
* RLS lets you **update θ incrementally** as new samples $(x_t, y_t)$ arrive, without reprocessing the whole dataset.

---

### Core idea

Maintain two things:

1. **Parameter vector** $\theta_t$: current regression coefficients.
2. **Inverse covariance matrix** $P_t = (X^\top X + \lambda I)^{-1}$: tracks “uncertainty” in estimates.

When a new sample $(x_t, y_t)$ comes in:

1. Compute **gain vector**:

   $$
   K_t = \frac{P_{t-1} x_t}{1 + x_t^\top P_{t-1} x_t}
   $$

2. Update coefficients:

   $$
   \theta_t = \theta_{t-1} + K_t \big(y_t - x_t^\top \theta_{t-1}\big)
   $$

   * This adjusts θ in the direction of the residual error, scaled by K.

3. Update covariance matrix:

   $$
   P_t = P_{t-1} - K_t x_t^\top P_{t-1}
   $$

   * Ensures future updates account for reduced uncertainty in direction $x_t$.

### Forgetting factor in RLS

In RLS with forgetting factor λ ∈ (0,1], the update equations are:

$$
\begin{aligned}
K_t &= P_{t-1} x_t \big/ \big(\lambda + x_t^\top P_{t-1} x_t\big) \\
\theta_t &= \theta_{t-1} + K_t (y_t - x_t^\top \theta_{t-1}) \\
P_t &= \frac{1}{\lambda}\big(P_{t-1} - K_t x_t^\top P_{t-1}\big)
\end{aligned}
$$

* λ acts on the **covariance matrix** P\_t, effectively down-weighting old observations.
* Smaller λ → faster adaptation to recent data, more “forgetting” of past.
* It’s mathematically more principled than the simple α blend, because it uses both past **information content** and new observation.

### Determine the Forgetting Factor

You can **map a desired moving window (e.g., 90 days) to a forgetting factor λ in RLS** using the exponential weighting interpretation. Here’s how it works step by step.

#### Exponential weighting in RLS

In RLS with forgetting factor λ, the **effective weight** of an observation from t days ago is roughly:

$$
w_t = \lambda^{\Delta t}
$$

* Δt = number of updates (or “days”) since that observation.
* Smaller λ → faster decay; λ close to 1 → slow decay (long memory).

So if you want a moving window of **N days**, you can set λ so that the weight of data N days ago is some small fraction, e.g., 0.05 or 5%:

$$
\lambda^N \approx 0.05
$$

---

#### Solve for λ

Example: N = 90 days

$$
\lambda = 0.05^{1/90} 
$$

Let’s calculate roughly:

$$
\ln(\lambda) = \frac{\ln(0.05)}{90} \approx \frac{-2.9957}{90} \approx -0.0333
$$

$$
\lambda \approx e^{-0.0333} \approx 0.967
$$

✅ So setting λ ≈ 0.967 will give the 90-day data an effective decay to \~5% weight.

#### Interpretation

* Any observation **older than 90 days** has minimal influence.
* Recent observations (last 10–30 days) dominate updates.
* You can adjust the “small fraction” (0.05) if you want a slightly shorter or longer effective memory.

---

#### Notes

* If you update **daily with batches**, each batch counts as one “update” in Δt.
* If you update **per impression**, Δt is per impression and you may need to adjust λ for the same effective window.

---


### How it works in Contextual Arms

* **Arm-specific RLS**
  For each arm $a$, maintain its own $\theta_a$ and $P_a$.
  Each time that arm is shown and you observe reward $r_t$:

  * Input $x_t$ (user context at decision time).
  * Target $y_t$ (observed reward).
  * Apply the update equations above.

* **Batch daily update**
  If you process logs in daily batches, just run RLS updates sequentially for each new sample in yesterday’s data.
  This is much faster than full retrain and equivalent to “online learning.”

* **Exploration & stability**

  * Keeps prior offline-trained θ as initialization (robust, low-variance).
  * Gradually adapts with new evidence.
  * If you want more responsiveness, introduce a **forgetting factor λ\_f** (0 < λ\_f ≤ 1):

    * Multiply P by $1/λ_f$ before each update.
    * This discounts old data → gives more weight to recent samples.

---

### Simple pseudocode (per arm)

```python
# Initialize from offline training
theta_a = theta_offline[a]         # coefficients
P_a = np.linalg.inv(XTX_offline[a] + lambda*np.eye(d))  

# Daily batch update
for (x, y) in yesterday_data_for_arm_a:
    x = x.reshape(-1,1)
    # Gain
    K = P_a @ x / (1 + x.T @ P_a @ x)
    # Update theta
    theta_a = theta_a + (K * (y - x.T @ theta_a)).flatten()
    # Update covariance
    P_a = P_a - K @ x.T @ P_a
```

* Store updated $\theta_a$ for next day’s scoring.
* Repeat for all arms.


