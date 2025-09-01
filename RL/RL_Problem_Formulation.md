
### 1. Common Elements of RL Problems

Reinforcement learning (RL) problems involve several common components that define the interaction between a learning entity and its environment, and how it seeks to achieve its goals.

Here are the common components of RL problems:

*   **Agent**
    *   The **learner and decision-maker** that interacts with the environment.
    *   It selects **actions** based on its perception of the environment's state.
    *   Its goal is to achieve an objective formalized by a reward signal.

*   **Environment**
    *   Comprises **everything outside the agent**.
    *   It responds to the agent's actions by presenting new situations (states) and emitting **reward signals**.
    *   The environment is often incompletely controllable and potentially incompletely known by the agent. The boundary between the agent and environment is typically drawn where the agent no longer has arbitrary control.

*   **States ($S_t$)**
    *   Signals conveyed to the agent that provide "some sense of 'how the environment is' at a particular time".
    *   States serve as the **basis for the agent's choices**.
    *   In Markov Decision Processes (MDPs), the formal framework for RL, **actions influence subsequent situations or states**.
    *   RL algorithms rely heavily on the concept of state as input to the policy and value function.

*   **Actions ($A_t$)**
    *   The **choices made by the agent** to influence its environment.
    *   Actions can be low-level controls (e.g., robot motor voltages) or high-level decisions (e.g., choosing where to drive).
    *   Actions affect the future state of the environment and thus the opportunities available to the agent at later times.

*   **Reward Signal ($R_t$)**
    *   A **special numerical signal** (a simple number, $R_t \in \mathbb{R}$) passed from the environment to the agent at each time step.
    *   It formalizes the agent's **purpose or goal**: to **maximize the total cumulative reward received over the long run**, not just immediate reward.
    *   The reward signal is considered external to the agent because it defines the task and is beyond the agent's arbitrary control. Rewards can be positive, negative, or zero.

*   **Policy ($\pi$)**
    *   Defines the learning agent's **way of behaving** at a given time.
    *   It is a **mapping from perceived states to probabilities of selecting each possible action** when in those states.
    *   The policy is the core of an RL agent, as it alone is sufficient to determine behavior. Policies can be stochastic, specifying probabilities for actions.
    *   RL methods specify how the agent's policy is changed as a result of its experience.

*   **Value Function**
    *   An estimate of "**how good**" it is for the agent to be in a given state, or to perform a given action in a given state.
    *   This "goodness" is defined in terms of the **expected future rewards**, or **expected return** ($G_t$).
    *   Value functions are central to most RL algorithms for **efficient search in the space of policies**.
    *   There are different types:
        *   **State-value function ($v_\pi(s)$)**: Expected return when starting in state $s$ and following policy $\pi$.
        *   **Action-value function ($q_\pi(s, a)$)**: Expected return when starting in state $s$, taking action $a$, and thereafter following policy $\pi$.
        *   **Optimal value functions ($v^*(s)$ and $q^*(s, a)$)**: The maximum expected return achievable from a state or state-action pair under any policy.

*   **Model of the Environment (Optional)**
    *   Something that **mimics the behavior of the environment**, allowing the agent to make inferences or predictions about how the environment will behave.
    *   For example, given a state and action, the model might predict the resultant next state and next reward.
    *   Models are used for **planning**, which involves deciding on a course of action by considering possible future situations before actually experiencing them.
    *   Methods using models for planning are called **model-based methods**, while those that learn by trial and error without a model are **model-free methods**.


The book "Reinforcement Learning: An Introduction" outlines various problem formulations and associated solution methods, evolving from simple non-associative tasks to complex, state-dependent problems, and then to challenges involving large state spaces and advanced techniques.

Here's a summary of the different problem formulations and methods.

### 2. Nonassociative Tasks (e.g., k-armed Bandit Problems)
*   **Problem Formulation**: In these tasks, there is **no need to associate different actions with different situations**. The learner either seeks a single best action in a stationary environment or tracks the best action in a nonstationary one. Each action affects only the **immediate reward**. 
    
    The agent does not deal with a changing environment state. Each action (arm pull) is independent of past actions, except for the information learned. Hence there's only a single state.

    An example is the **k-armed bandit problem**, where an agent repeatedly chooses among *k* options, receiving a numerical reward for each choice, with the objective of **maximizing the expected total reward** over time.
*   **Goal**: Maximize expected total reward.
*   **Methods**:
    *   **Action-value methods**: Estimate the value of each action by averaging the rewards received when that action is selected.
    *   Policy = Action selection strategy: **Balancing Exploration and Exploitation**
        *   **$\epsilon$-greedy methods**: Choose a random action a small fraction of the time, otherwise exploit the action with the highest estimated value.
        *   **Optimistic initial values**: Initialize action-value estimates optimistically to encourage early exploration.
        *   **Upper-Confidence-Bound (UCB) Action Selection**: Selects actions by subtly favoring those that have received fewer samples to ensure exploration.
        *   **Gradient Bandit Algorithms**: Estimate action preferences rather than values, favoring more preferred actions probabilistically using a softmax distribution.
*   **Associative Search (Contextual Bandits)**: An intermediate problem where the best action depends on the situation, requiring learning a policy (a mapping from situations to best actions). The agent uses this context to choose an action.
    - Context ($x_i$) 
        - Like a "state", but unlike the state definition in RL, it **doesn’t evolve based on the agent’s action**.
        - Example: user profile in recommendation systems, or patient features in medical treatment.
    - Goal: Learn a policy $\pi(a|x)$ that maximizes expected reward conditioned on context.

### 3. Finite Markov Decision Processes (MDPs)
*   **Problem Formulation**: This is the **formal framework for reinforcement learning**. It involves **evaluative feedback** and an **associative aspect**, where actions influence not just immediate rewards but also **subsequent situations (states)** and **future rewards**, introducing the concept of **delayed reward**. The environment is often **incompletely controllable and potentially incompletely known**. The core elements are:
    *   **States ($S_t$)**: Information about the environment that serves as the basis for the agent's choices.
    *   **Actions ($A_t$)**: Choices made by the agent to influence the environment.
    *   **Reward Signal ($R_t$)**: A numerical signal defining the agent's goal: **maximize total cumulative reward over the long run**.
    *   **Policy ($\pi$)**: The agent's behavior, a mapping from perceived states to actions.
    *   **Value Functions ($v_\pi(s)$ and $q_\pi(s,a)$)**: Estimates of "how good" a state or state-action pair is, in terms of **expected future rewards (expected return)**.
    *   **Returns ($G_t$)**: The function of future rewards that the agent seeks to maximize, which can be **discounted** or **undiscounted**.
*   **Task Types**:
    *   **Episodic tasks**: Agent-environment interaction naturally breaks into a sequence of separate episodes.
    *   **Continuing tasks**: Interaction does not naturally terminate but continues indefinitely.
*   **Levels of Knowledge**:
    *   **Complete knowledge**: Agent has a complete and accurate model of the environment's dynamics, i.e., transition probabilities $p(s', r|s,a)$.
    *   **Incomplete knowledge**: A complete and perfect model is not available.

*   **Methods for Complete Knowledge (Dynamic Programming - DP)**:
    *   **Dynamic Programming (DP)**: A collection of algorithms that compute optimal policies given a perfect model of the environment as an MDP. DP relies on **bootstrapping**, updating estimates based on other estimates.
        *   **Policy Evaluation (Prediction)**: Iteratively computes the value function for a given policy.
        *   **Policy Improvement**: Computes an improved policy based on a given value function.
        *   **Policy Iteration**: Alternates between policy evaluation and policy improvement until an optimal policy is found.
        *   **Value Iteration**: Combines policy evaluation and policy improvement into a single iterative step, converging to the optimal value function.
        *   **Asynchronous DP methods**: Update states in an arbitrary order rather than sweeping through the entire state set.

*   **Methods for Incomplete Knowledge (Model-Free Learning)**:
    *   **Monte Carlo (MC) Methods**: Learn value functions and optimal policies directly from **sample episodes (experience)** without requiring a model of the environment's dynamics. They involve averaging returns from many random samples.
        *   **MC Prediction**: Estimates $v_\pi(s)$ or $q_\pi(s,a)$ by averaging returns from visits to states.
            *   **First-visit MC**: Averages returns only after the first visit to a state in an episode.
            *   **Every-visit MC**: Averages returns after every visit to a state in an episode.
        *   **MC Control**: Uses **Generalized Policy Iteration (GPI)**, where policies are improved with respect to learned value functions.
            *   **Exploring Starts (ES)**: Ensures that all state-action pairs are visited infinitely often by starting episodes in randomly chosen state-action pairs.
            *   **On-policy MC Control**: Learns about the policy currently being used to generate behavior.
            *   **Off-policy MC Control**: Uses a **behavior policy** to generate experience and a **target policy** (the one being learned about) to evaluate actions. This typically involves **importance sampling** to weight returns.
                *   **Ordinary Importance Sampling**: Unbiased but can have high variance.
                *   **Weighted Importance Sampling**: Biased but generally has lower variance and is preferred in practice.
    *   **Temporal-Difference (TD) Learning Methods**: A **combination of Monte Carlo and Dynamic Programming ideas**. They learn directly from experience without a model but update estimates based on other learned estimates (**bootstrapping**) without waiting for a final outcome.
        *   **TD Prediction**: Estimates $v_\pi(s)$ for a given policy $\pi$.
            *   **One-step TD (TD(0))**: Updates estimates based on the immediate reward and the estimated value of the next state.
        *   **TD Control**: Extends TD prediction using GPI.
            *   **On-policy TD Control**:
                *   **Sarsa**: Learns an action-value function $q_\pi(s,a)$ for the current policy by taking action $A_t$, observing $R_{t+1}, S_{t+1}$, then taking action $A_{t+1}$ before updating.
            *   **Off-policy TD Control**:
                *   **Q-learning**: Learns the optimal action-value function $q^*(s,a)$ by updating towards the maximum possible reward in the next state, independent of the actual action taken by the behavior policy.
                *   **Expected Sarsa**: Similar to Q-learning but updates towards the expected value of the next state-action pair under the target policy.
                *   **Double Learning (e.g., Double Q-learning, Double Expected Sarsa)**: Addresses maximization bias by maintaining two independent action-value estimates.
            *   **Afterstates**: In some problems (e.g., games), the immediate effect of an action is fully known, allowing value functions to be defined over "afterstates" (states immediately following an action).

### 4. Unifying Monte Carlo and TD Methods (n-step Bootstrapping & Eligibility Traces)
*   **Problem Formulation**: Addresses the trade-off between bias (from bootstrapping) and variance (from long returns) by allowing intermediate degrees of bootstrapping.
*   **Methods**:
    *   **n-step TD Methods**: Generalize one-step TD and Monte Carlo by basing updates on *n* steps of observed rewards and the estimated value of the state *n* steps later.
        *   **n-step Sarsa**: On-policy control using n-step returns for action values.
        *   **n-step Expected Sarsa**: Off-policy control with expected value for the last step.
        *   **n-step Tree Backup Algorithm**: Off-policy method that does not use importance sampling.
        *   **n-step Q($\sigma$)**: A unifying algorithm that allows choosing, on a step-by-step basis, whether to sample an action (like Sarsa) or consider the expectation over all actions (like Tree Backup).
    *   **Eligibility Traces**: Provide an **elegant algorithmic mechanism with significant computational advantages** over n-step methods, unifying TD and Monte Carlo in a more general way. They use a **short-term memory vector (eligibility trace $z_t$)** to give credit to recently visited states or state-action pairs.
        *   **$\lambda$-return Algorithms**: Allow continuous shifting between one-step TD ($\lambda=0$) and Monte Carlo ($\lambda=1$).
        *   **Sarsa($\lambda$)**: An on-policy action-value method that uses eligibility traces.
        *   **Off-policy Eligibility Traces**: Incorporate importance sampling and control variates to extend eligibility traces to off-policy learning.
        *   **Watkins's Q($\lambda$)**: An early off-policy method that cuts eligibility traces after non-greedy actions.
        *   **Tree-Backup($\lambda$)**: Eligibility trace version of Tree Backup, which avoids importance sampling.
        *   **Stable Off-policy methods with traces (e.g., Gradient-TD, Emphatic-TD)**: Address the "deadly triad" (function approximation, off-policy learning, bootstrapping) to guarantee convergence.

### 5. Planning and Learning
*   **Problem Formulation**: Integrates **model-based methods (planning)** and **model-free methods (learning)**.
*   **Models**:
    *   **Distribution Model**: Provides full probabilities of next states and rewards (required by DP).
    *   **Sample Model**: Generates single sample transitions and rewards (needed for simulation).
*   **Planning**: Any computational process that takes a model as input and produces or improves a policy.
    *   **State-space planning**: Searches through the state space for an optimal policy or path.
*   **Methods**:
    *   **Dyna Architecture (Dyna-Q)**: A simple architecture that **integrates direct reinforcement learning, model-learning, and planning**. It uses real experience for direct learning and model updates, and simulated experience from the model for planning.
    *   **Prioritized Sweeping**: Focuses planning computations by working backward from states whose values have significantly changed, updating their predecessors.
    *   **Trajectory Sampling**: Focuses updates on states or state-action pairs that are likely to be visited under the current policy.
    *   **Real-time Dynamic Programming (RTDP)**: An on-policy trajectory-sampling version of value iteration, focusing computation on relevant states reachable from a starting point.
    *   **Decision-time Planning (e.g., Heuristic Search, Rollout Algorithms, Monte Carlo Tree Search)**: Planning initiated and completed after encountering each new state to select an immediate action, often involving deep lookaheads.

### 6. Approximate Solution Methods (with Function Approximation)
*   **Problem Formulation**: Applicable to problems with **arbitrarily large state spaces** (e.g., those with continuous states or features) where tabular methods are infeasible. The goal is to find a **good approximate solution** using limited resources, rather than an exact optimal one. This involves **generalization from examples**.
*   **Function Approximation Methods**:
    *   **Parameterized Function Approximation**: Represents the value function (or policy) as a functional form with a finite-dimensional **weight vector $\mathbf{w}$**. Changes in weights generalize across states.
        *   **Linear Methods**: The approximate value is a linear function of the weight vector, $v̂(s,\mathbf{w}) = \mathbf{w}^T \mathbf{x}(s)$, where $\mathbf{x}(s)$ is a **feature vector** representing state *s*.
            *   **Feature Construction**: Various bases can be used, such as **polynomials**, **Fourier basis**, **coarse coding**, **tile coding (CMAC)**, and **radial basis functions (RBFs)**.
        *   **Artificial Neural Networks (ANNs)**: Multi-layer ANNs can compute the value function, with weights as parameters.
        *   **Decision Trees**: Can also be used as function approximators.
    *   **Memory-based Function Approximation**: Stores training examples and retrieves a subset (e.g., nearest neighbors) to estimate the value of a query state. Examples include **nearest neighbor**, **weighted average**, and **locally weighted regression**.
    *   **Kernel-based Function Approximation**: Uses a **kernel function** to assign weights based on similarity between states.
*   **Learning Objectives**:
    *   **Mean Square Value Error (VE)**: A common measure of approximation error under the on-policy state distribution.
*   **Methods with Approximation**:
    *   **On-policy Prediction with Approximation**: Estimates $v_\pi$ using parameterized functions.
        *   **Semi-gradient TD Methods**: Adjust weights using a gradient-descent-like update, but treat the update target as fixed (ignoring its dependence on weights for the gradient calculation).
            *   **Semi-gradient TD(0)**.
            *   **Gradient Monte Carlo**.
            *   **n-step Semi-gradient TD**.
        *   **Interest and Emphasis**: Mechanisms to weight the importance of accurately valuing specific states, focusing approximation resources.
    *   **On-policy Control with Approximation**:
        *   **Episodic Semi-gradient Sarsa**: Extends Sarsa to use function approximation.
        *   **Average Reward Setting**: For continuing tasks without discounting, using **differential returns** and **differential value functions**. **Differential semi-gradient Sarsa** is an algorithm for this setting.
    *   **Off-policy Methods with Approximation**: More challenging due to potential for **instability and divergence** (the "deadly triad" of function approximation, off-policy learning, and bootstrapping).
        *   **Semi-gradient Off-policy TD methods**: Straightforward extensions of tabular methods, but may diverge.
        *   **Gradient-TD Methods**: True stochastic gradient descent methods that minimize the projected Bellman error (PBE), offering robust convergence for linear function approximation, albeit with increased computational complexity.
        *   **Emphatic-TD Methods**: Reweight updates to restore desirable convergence properties of on-policy learning, ensuring stability.

### 7. Policy Gradient Methods
*   **Problem Formulation**: Instead of learning action values to derive a policy, these methods **directly learn a parameterized policy** that selects actions without necessarily consulting a value function.
*   **Goal**: Maximize a scalar performance measure $J(\theta)$ with respect to the policy parameter $\theta$.
*   **Methods**:
    *   **REINFORCE**: A Monte Carlo policy-gradient control method that updates policy parameters in the direction of an estimate of the gradient of performance.
    *   **REINFORCE with Baseline**: Reduces the variance of REINFORCE updates by subtracting a baseline (often a learned state-value function) without introducing bias.
    *   **Actor-Critic Methods**: Use a learned state-value function (the **critic**) to assess actions, which helps reduce variance in the policy gradient (the **actor**). They introduce some bias but improve learning efficiency. Can be one-step or incorporate eligibility traces.
    *   **Policy Parameterization for Continuous Actions**: Policies can be parameterized as probability distributions (e.g., Gaussian) over real-valued actions, with the mean and standard deviation approximated by functions of the state.
    *   **Policy Gradient for Continuing Problems**: Adapts to the average reward setting, using differential values.

### 8. Frontiers and Advanced Concepts
*   **General Value Functions (GVFs)**: Generalize value functions beyond reward prediction to predict arbitrary **cumulants** (any signal) over state-dependent horizons. GVFs can serve as **auxiliary tasks** to improve representation learning and overall performance.
*   **Temporal Abstraction via Options**: Formalizes **temporally extended courses of action** as "options," which have their own policies and termination conditions. This allows planning and learning at multiple time scales.
*   **Observations and State**: Addresses challenges of **partial observability**, where the agent's sensory input only provides partial information about the true environment state. This involves constructing states from histories of observations and actions.
*   **Designing Reward Signals**: Explores techniques like **reward shaping**, where the reward signal is gradually modified as learning proceeds to guide the agent in tasks with sparse rewards.

These formulations and methods represent the core ideas and approaches within reinforcement learning, ranging from foundational concepts to advanced techniques for addressing complex, real-world challenges.