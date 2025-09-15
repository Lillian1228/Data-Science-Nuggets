

Here’s a structured extraction of the **DeepHit project (AAAI 2018)** based on a machine learning system design framework:

---

### 1. Business Problem, Objective, and Constraints

* **Problem**: Survival analysis (time-to-event prediction), especially under *competing risks* (e.g., cancer vs. cardiovascular death). Traditional models rely on strong, often unrealistic parametric assumptions.
* **Objective**: Learn the joint distribution of survival times and event types directly from data, without restrictive assumptions, and improve predictive performance in both single-risk and competing-risk settings.
* **Constraints**:

  * Right-censoring is common in medical data (patients lost to follow-up).
  * Need to capture non-linear, time-varying relationships between covariates and risks.
  * Model must be robust across diverse datasets (clinical, genomic, synthetic).

---

### 2. Frame the Problem into an ML Task

* **ML Task**: Probabilistic prediction task – estimate $P(s, k \mid x)$, the probability that an event of type $k$ occurs at time $s$, given covariates $x$.
* This is a **multi-task learning** problem (multiple event types/risks), combined with structured output (time + event).
* Evaluation: **time-dependent concordance index (Ctd)** to measure risk ranking over time.

---

### 3. What Methods Have Been Considered and Why Deep Learning

* **Traditional methods**:

  * Cox Proportional Hazards (CPH): assumes proportional hazards and linear covariate-risk relationships.
  * Fine-Gray model: handles competing risks but assumes proportional hazard rates.
  * Random survival forests, Bayesian Gaussian processes, dependent logistic regressors: flexible but weak in competing risk settings.
* **Why Deep Learning**:

  * Can learn *non-linear* and *time-varying* covariate effects.
  * Can jointly model multiple risks without assuming independence.
  * Multi-task architecture captures shared and cause-specific representations.
  * Outperforms prior methods on synthetic and real datasets.

---

### 4. Data Preparation

* **Datasets**:

  * **SEER**: 68k breast cancer patients, 2 competing risks (cancer vs. CVD).
  * **UNOS**: 60k heart transplant patients, single risk (death).
  * **METABRIC**: 1.9k breast cancer patients, single risk, with genomic & clinical features.
  * **Synthetic**: constructed with non-linear quadratic covariate relationships and censoring.
* **Preprocessing**:

  * Missing values imputed (mean for continuous, mode for categorical).
  * One-hot encoding for categorical variables.
  * Features standardized.
  * Survival time discretized, finite time horizon assumed.

---

### 5. Model Training and Parameter Tuning

* **Training**:

  * Loss = $L_{Total} = L_1 + L_2$:

    * $L_1$: log-likelihood accounting for right-censoring.
    * $L_2$: ranking loss inspired by concordance index, enforces correct risk ordering.
  * Optimizer: Adam, learning rate $10^{-4}$.
  * Batch size: 50.
  * Dropout: 0.6.
  * Xavier initialization.
  * Early stopping on validation loss.
* **Hyperparameters**:

  * $\alpha$ (trade-off between losses).
  * $\sigma$ (ranking loss sharpness).
  * Selected via validation set performance.

---

### 6. Model Architecture Design

* **DeepHit Architecture**:

  * **Shared sub-network**: fully-connected layers, extracts common latent representation from covariates.
  * **Cause-specific sub-networks**: separate branches for each event type, conditioned on both shared features and raw covariates (residual connection).
  * **Output layer**: single softmax producing joint distribution over time × event.
  * **Layers**:

    * Shared: 1 fully-connected layer.
    * Cause-specific: 2 fully-connected layers each.
    * Hidden dimensions = multiples of input covariate size (3×, 5×, 3×).
    * Activation: ReLU.

---

### 7. Model Serving and Implementation (if any)

* Implemented in **TensorFlow**.
* Cross-validation setup: 5-fold CV, with validation split from training data.
* Serving/production deployment not discussed in the paper (research-focused).
* Practical implication: can be used for **personalized risk prediction** in medicine (e.g., informing treatment choices under competing risks).

---

### 8. Limitations and Future Areas for Improvements

* **Limitations**:

  * Assumes discrete time survival; may lose information vs. continuous-time models.
  * Only considers one event per patient; does not yet handle recurrent/multiple sequential events.
  * Computationally more expensive than traditional survival models.
* **Future improvements**:

  * Extend to multi-event settings where several diseases can occur (not just one competing risk).
  * Improve interpretability for clinical use (explainable risk factors).
  * Explore scalability to very high-dimensional genomic/omics data.
  * Better handling of censoring assumptions (currently assumes censoring at random).
  * Potential for integration into real-time clinical decision support systems.

---

Would you like me to also **visualize this framework as a one-page diagram** (ML system design map of DeepHit), so it’s easier to digest for presentations or notes?
