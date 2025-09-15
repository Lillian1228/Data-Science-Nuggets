## Single Neuron with Sigmoid Activation and Backpropagation

This task involves implementing backpropagation for a single neuron in a neural network. The neuron processes inputs and updates parameters to minimize the Mean Squared Error (MSE) between predicted outputs and true labels.

### Mathematical Background

**Forward Pass**  
Compute the neuron output by calculating the dot product of the weights and input features, and adding the bias:
$$
z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
$$
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Loss Calculation (MSE)**  
The Mean Squared Error quantifies the error between the neuron's predictions and the actual labels:
$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (\sigma(z_i) - y_i)^2
$$

### Backward Pass (Gradient Calculation)
Compute the gradient of the MSE with respect to each weight and the bias. This involves the partial derivatives of the loss function with respect to the output of the neuron, multiplied by the derivative of the sigmoid function:
$$
\frac{\partial MSE}{\partial w_j} = \frac{2}{n} \sum_{i=1}^{n} (\sigma(z_i) - y_i) \sigma'(z_i) x_{ij}
$$
$$
\frac{\partial MSE}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (\sigma(z_i) - y_i) \sigma'(z_i)
$$

### Parameter Update
Update each weight and the bias by subtracting a portion of the gradient, determined by the learning rate:
$$
w_j = w_j - \alpha \frac{\partial MSE}{\partial w_j}
$$
$$
b = b - \alpha \frac{\partial MSE}{\partial b}
$$

### Practical Implementation
This process refines the neuron's ability to predict accurately by iteratively adjusting the weights and bias based on the error gradients, optimizing the neural network's performance over multiple iterations.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs):
    """Write a Python function that simulates a single neuron with sigmoid activation, and implements backpropagation to update the neuron's weights and bias. """
    weights = np.array(initial_weights)
    bias = initial_bias
    features = np.array(features)
    labels = np.array(labels)
    mse_values = []

    for _ in range(epochs):
        z = np.dot(features, weights) + bias
        predictions = sigmoid(z)
        
        mse = np.mean((predictions - labels) ** 2)
        mse_values.append(round(mse, 4))

        # Gradient calculation for weights and bias
        errors = predictions - labels
        weight_gradients = (2/len(labels)) * np.dot(features.T, errors * predictions * (1 - predictions))
        bias_gradient = (2/len(labels)) * np.sum(errors * predictions * (1 - predictions))
        
        # Update weights and bias
        weights -= learning_rate * weight_gradients
        bias -= learning_rate * bias_gradient

        # Round weights and bias for output
        updated_weights = np.round(weights, 4)
        updated_bias = round(bias, 4)

    return updated_weights.tolist(), updated_bias, mse_values
```