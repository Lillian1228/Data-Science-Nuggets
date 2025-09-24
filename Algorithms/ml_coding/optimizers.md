
## Stage 1. Vanilla Gradient Descent

Gradient Descent is an optimization algorithm used to minimize the cost function in machine learning models, particularly in linear regression and neural networks. 

**Update rule:**

$$
\theta_{t+1} = \theta_t - \eta \, \nabla L(\theta_t)
$$

* **Pros:** Simple, foundational.
* **Cons:** Needs careful learning rate tuning; slow for large-scale, non-convex problems; oscillates in ravines.

There are three main types of gradient descent based on how much data is used to compute the gradient at each iteration. Using Mean Squared Error (MSE) loss function as an example:

- Batch Gradient Descent: Batch Gradient Descent computes the gradient of the MSE loss function with respect to the parameters **for the entire training dataset**. It updates the parameters after processing the entire dataset:
    $$
    \theta = \theta - \alpha \cdot \frac{2}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}
    $$
    where $ \alpha $ is the learning rate, $ m $ is the number of samples, and $ \nabla_{\theta} J(\theta) $ is the gradient of the MSE loss function.
- Stochastic Gradient Descent (SGD):
Stochastic Gradient Descent updates the parameters **for each training example** individually, making it faster but more noisy:
    $$
    \theta = \theta - \alpha \cdot 2 \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}
    $$
    where $ x^{(i)}, y^{(i)} $ are individual training examples.
- Mini-Batch Gradient Descent:
Mini-Batch Gradient Descent is a compromise between Batch and Stochastic Gradient Descent. It updates the parameters after processing a small batch of training examples, without shuffling the data:
    $$
    \theta = \theta - \alpha \cdot \frac{2}{b} \sum_{i=1}^{b} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}
    $$
    where $ b $ is the batch size, a subset of the training dataset.

## Stage 2. Momentum Based

### 2.1 **SGD with Momentum (Polyak, 1964)**

Momentum is a popular optimization technique that helps **accelerate gradient descent in the relevant direction and dampens oscillations**. It works by **adding a fraction of the previous update vector** to the current gradient. It uses a moving average of gradients to determine the direction of the update. 

**Update rule:**

$v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta)$ (Velocity update)

$\theta_t = \theta_{t-1} - v_t$ (Parameter update)

Where:
- $v_t$ is the velocity at time t
- $\gamma$ is the momentum coefficient (typically 0.9)
- $\eta$ is the learning rate
- $\nabla_\theta J(\theta)$ is the gradient of the loss function


**Pros:** Faster convergence, reduces oscillations, helps escape shallow minima.

**Cons:** Adds a hyperparameter ($\beta$); still sensitive to learning rate.

### 2.2 **Nesterov Accelerated Gradient (NAG, 1983; popularized \~2013)**

Nesterov Accelerated Gradient (NAG) is an improvement over classical momentum optimization. While momentum helps accelerate gradient descent in the relevant direction, NAG takes this a step further by **looking ahead** in the direction of the momentum before computing the gradient. This "look-ahead" property helps NAG make more informed updates and often leads to better convergence.

Nesterov Accelerated Gradient uses a "look-ahead" approach where it first makes a momentum-based step and then computes the gradient at that position. 

**Update rule:**

$\theta_{lookahead, t-1} = \theta_{t-1} - \gamma v_{t-1}$ (Look-ahead position)

$v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta_{lookahead, t-1})$ (Velocity update)

$\theta_t = \theta_{t-1} - v_t$ (Parameter update)

Where:
- $v_t$ is the velocity at time t
- $\gamma$ is the momentum coefficient (typically 0.9)
- $\eta$ is the learning rate
- $\nabla_\theta J(\theta)$ is the gradient of the loss function

The key difference from classical momentum is that the gradient is evaluated at $\theta_{lookahead, t-1}$ instead of $\theta_{t-1}$

**Pros:** “Looks ahead” → better convergence than plain momentum.

**Cons:** More complex, tuning still required.

### Stage 3. Adaptive Learning Rates

### 3.1 **Adagrad (2011)**

Adagrad accumulates the squared gradients for each parameter from history, and adapts the learning rate to each parameter so that it performs larger updates for infrequent parameters and smaller updates for frequent ones. This makes it particularly well-suited for dealing with sparse data.

**Update rule:**

$G_t = G_{t-1} + g_t^2$ (Accumulated squared gradients)

$\theta_t = \theta_{t-1} - \dfrac{\alpha}{\sqrt{G_t} + \epsilon} \cdot g_t$ (Parameter update)

Where:
- $G_t$ is the sum of squared gradients up to time step t
- $\alpha$ is the initial learning rate
- $\epsilon$ is a small constant for numerical stability
- $g_t$ is the gradient at time step t


**Pros:** Per-parameter learning rate; good for sparse data (e.g., NLP with rare words).

**Cons:** Learning rate decays too aggressively as we do not "forget" historical gradients → training stalls.

### 3.2 **RMSProp (2012, Hinton’s lecture)**

RMSProp introduces exponential moving average of squared gradients based on Adagrad, to fix the decay problem in Adagrad where large cumulative gradients in later iterations can cause learning rate to be too small. It allows "forgetting"/downweighting gradients cumulated from earlier iterations.

$$
E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta) \nabla L(\theta_t)^2, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \nabla L(\theta_t)
$$

* **Pros:** Fixes Adagrad’s decay problem; stable for RNNs.
* **Cons:** No momentum originally (though often combined).

## Stage 4: Adam (Adaptive Moment Estimation) Family

### 4.1 **Adam (Kingma & Ba, 2014) = RMSProp + Momentum**

Adam combines the benefits of two other optimization algorithms: RMSprop and momentum optimization.

Adam maintains moving averages of both gradients (first moment) and squared gradients (second moment) to adapt the learning rate for each parameter. 

**Update rule:**

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L(\theta_t) \quad \text{(first moment, same as momemtum)}  
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \quad \text{(second moment, same as RMSPop)}  
$$

- Direction: the first-order momentum can smooth the gradient and avoid oscillations. The average points more consistently toward the long-term downhill direction, instead of being swayed by short-term noise.
- Scale: the second-order momentum rescales each parameter’s step size (learning rate), preventing updates that are too large or too small in certain directions.
    - Small step if past gradients are large (steep slope).
     - Large step if past gradients are small (flat slope).

Bias correction:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

- Why? 

    At the start of training, both $m_0$ and $v_0$ are initialized to zero, so moving avgs are baised towards zero as they haven't seen enough history. The denominator $1−β^t$ compensates for the fact that at small t, the moving averages are shrunken toward zero. As t &rarr; $\inf$, $1−β^t$ &rarr; 1 and the correction fades away.

Parameter update:
$$
\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$


**Pros:** Combines momentum + adaptive rates; works well “out of the box”; robust.

- Momentum → smooth path downhill.
- Adaptive scaling → right step size in every direction.
- Bias correction → stable from the start.

**Cons:** Sometimes worse generalization than SGD; still needs LR tuning.

```Python
import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
	"""
    grad: A function that computes the gradient of f
    """
	x = x0
    
    # vector initialization
    m=np.zeros_like(x)
    v=np.zeros_like(x)

    for t in range(1, num_iterations+1):

        g = grad(x)

        # moments update
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*(g**2)
        # bias correction is decoupled from m,v update to avoid overwrite
        m_hat = m/(1-beta1**t) 
        v_hat = v/(1-beta2**t)
        # parameter update
        x = x - learning_rate*m_hat/(np.sqrt(v_hat)+epsilon)

    return x
```

### 4.2 **AdamW (2017)**

**Motivation**: 

In order to apply regularization with Adam, people used to just **add the L2 term into the gradient itself**:

$$
g_t = \nabla L(\theta_t) + \lambda \theta_t
$$

Then the rest of Adam’s update (momentum, scaling, bias correction) would use this modified gradient. This raises problems:

* When you mix $\lambda \theta_t$ into $g_t$, the weight decay gets entangled with Adam’s adaptive scaling.
* That means weight decay no longer acts like true L2 shrinkage.

AdamW fixes the poor regularization problem by **decoupling weight decay** from gradient update. It keeps:

$$
g_t = \nabla L(\theta_t)
$$

Then after the Adam update, we apply weight decay separately:

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} - \boxed{\eta \lambda \theta_t}
$$

So **AdamW never redefines $g_t$** — it only decouples the shrinkage from the gradient.


**Pros:** Fixes Adam’s poor regularization; now the default in vision and transformers.

**Cons:** Still more complex than SGD.

```Python
import numpy as np

def adamw_update(w, g, m, v, t, lr, beta1, beta2, epsilon, weight_decay):
    m_new = beta1 * m + (1 - beta1) * g
    v_new = beta2 * v + (1 - beta2) * (g ** 2)
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)
    w = w - lr * weight_decay * w  # decoupled weight decay
    w_new = w - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return w_new, m_new, v_new
```

### **Nadam (2016)**

* Adam + Nesterov momentum.
* **Pros:** Slightly better convergence in some cases.
* **Cons:** Gains over Adam are modest.

---

## 🔹 Stage 5: Beyond Adam (Modern refinements)

* **AdaMax** (2014): Adam with infinity norm; stable in some edge cases.
* **RAdam (2019)**: Rectified Adam; adds variance correction to stabilize training.
* **LAMB / LARS** (2019): Layer-wise scaling; used for very large batch training (e.g., BERT, GPT).



| Optimizer        | Gradient Update                                 | Pros                       | Cons                               |
| ---------------- | ----------------------------------------------- | -------------------------- | ---------------------------------- |
| **GD / SGD**     | Fixed global step in gradient direction         | Simple, foundational       | Needs LR tuning, slow              |
| **Momentum**     | Adds velocity term                              | Faster, smoother           | Still sensitive to LR              |
| **NAG**          | Momentum + lookahead                            | More accurate steps        | More complex                       |
| **Adagrad**      | Scales LR by past gradients (cumulative)        | Good for sparse data       | LR decays too fast                 |
| **RMSProp**      | Exponential moving average of squared gradients | Fixes Adagrad, stable      | Still requires LR tuning           |
| **Adam**         | Momentum + RMSProp                              | Great default, adaptive    | Can overfit, weaker generalization |
| **AdamW**        | Adam + proper weight decay                      | Best default for modern DL | More tuning knobs                  |
| **Nadam**        | Adam + Nesterov                                 | Sometimes faster           | Marginal gains                     |
| **RAdam / LAMB** | Adam refinements for stability & scale          | Scales to huge models      | Niche use cases                    |

---

👉 Evolution in one line:
**SGD → add momentum (faster) → add adaptivity (Adagrad, RMSProp) → combine both (Adam) → refine regularization & scaling (AdamW, LAMB).**

