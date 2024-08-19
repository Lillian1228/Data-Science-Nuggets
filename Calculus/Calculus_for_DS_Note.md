
![overview](src/overview.png)

### 1. Derivatives

Derivatives are used to maximize or minimize functions, which is crucial when finding the best model to fit data. 

#### 1.1 Intuition/Analogy

**Average velocity** is the total distance traveled divided by the total time taken, while **instantaneous velocity** is the velocity at a specific moment in time.

Given a table of values showing the distance traveled every five seconds for one minute, we can calculate the average velocity in a given interval, b ut we cannot determine the velocity at a specific instant without more refined measurements. **By taking even finer intervals, we can further improve our estimate of the velocity at a specific instant.** This leads us to the concept of the derivative.

#### 1.2 Derivatives and Tangents

The **instantaneous rate of change**, also known as the derivative, measures how fast the relation between two variables is changing at any point. The derivative of a function at a point is precisely the **slope of the tangent line at that point ($dx/dt$)**.

![derivative0](src/derivative0.png)

When finding the maximum or minimum in a function, these points occur where the derivative is zero or where the tangent line is horizontal.

![derivative0](src/derivative1.png)

#### 1.3 Some Common Derivatives

1. Lines: For f(x) = ax + b, f'(x) = a, where a is the slope and b is the y-intercept.

2. Quadratics: 
![derivative0](src/derivative2.png)

3. Higher degree polynomials:
![derivative0](src/derivative3.png)

4. Other power functions:
![derivative0](src/derivative4.png)

For a function $f(x) = x^n$, where n is a constant exponent, then the derivative of this function is given by: $f'(x) = n*x^{n-1}$

5. Trigonometric functions:
   
   For $f(x) = sin(x)$, $f'(x) = cos(x)$  
   For $f(x) = cos(x)$, $f'(x) = -sin(x)$ 



#### 1.4 Inverse Function and its Derivative

Let f be a function that maps elements from a set A to a set B. The inverse function of f, denoted as $f^{-1}$, is a function that maps elements from set B back to set A, such that for every element y in set B, if f(x) = y, then $f^{-1}(y) = x$. 

In other words, the inverse function undoes the action of the original function, allowing us to retrieve the original input from the output.

![derivative0](src/derivative5.png)

#### 1.5 Exponential and logarithm functions

**Euler's number e** is a special number in mathematics that appears in various branches and can be defined in different ways. One way to define it is by the numerical value 2.71828182, etc. 

Another way is using the expression $(1+1/n)^n$. The more we increase n, the closer this expression converges to the number e, which is approximately 2.71828182. One example of n is the number of times a bank assign interests.

![derivative0](src/derivative6.png)

- If $f(x) = e^x$, then $f'(x) = e^x$. The derivative of the exponential function is itself.

The **logarithm** of a number x is the exponent to which the base e (Euler's number) must be raised to obtain x. It is denoted as log(x) or ln(x) for the natural logarithm.

The exponential function $e^x$ and the logarithmic function $log(x)$ are inverses of each other. This means that if we reflect one function about the line y = x, we get the other function.
- If $f(x) = log(x)$, then $f'(x) = 1/x$.
  
![derivative0](src/derivative7.png)

#### 1.6 Existence of derivatives

A differentiable function is a function that **has a derivative at every point** in its domain. In other words, a differentiable function is one for which you **can find the slope of the tangent line at any point on the function's graph**. If a function is differentiable, it means that it is smooth and has no abrupt changes or discontinuities.

Examples of different types of non-differentiable functions:

1. functions with corners or cusps
![derivative0](src/derivative8.png)

2. functions with jump discontinuities
![derivative0](src/derivative9.png)
3. functions with vertical tangents (vertical has no well-defined slope, rise can be any number while run is 0)
![derivative0](src/derivative10.png)

#### 1.7 Properties of the derivative

1. Scalar multiplication: 
   if $f=cg$, where c is a constant, then $f'=cg'$.
2. Sum rule:
   if $f=g+h$, then $f'=g'+h'$.

3. Product rule (like hitting a hammer on each element):
   if $f=gh$, then $f'=g'h+gh'$.
4. Chain rule based on two notations:
   ![derivative0](src/derivative11.png)
   e.g. If $f(x) = e^{2x}$, then $f(x) = g(2x)$, $f'(x)=2e^{2x}$. 

#### 1.8 Differentiation Methods in Python
1. Symbolic Differentiation with SymPy: 
   
   Symbolic computation deals with the computation of mathematical objects that are represented exactly, not approximately. For differentiation it would mean that the output will be somehow similar to if you were computing derivatives by hand using rules (analytically). Thus, symbolic differentiation can produce exact derivatives. 
   
   Cons - unefficiently slow computations and very complicated function as an output (expression swell) when there is a "jump" in the derivative

2.  Numerical Differentiation with NumPy :  
   This method does not take into account the function expression. The only important thing is that the function can be evaluated in the nearby points $x$ and $x+\Delta x$, where $\Delta x$ is sufficiently small. Then $\frac{df}{dx}\approx\frac{f\left(x + \Delta x\right) - f\left(x\right)}{\Delta x}$, which can be called a **numerical approximation** of the derivative.  
   Cons -  1) inaccurate at the points where there are "jumps" of the derivative; 2)  slow speed and requires function evalutation every time.

3. Automatic Differentiation:  
   It breaks down the function into common functions ( 𝑠𝑖𝑛,  𝑐𝑜𝑠,  𝑙𝑜𝑔, power functions, etc.), and constructs the computational graph consisting of the basic functions. Then the chain rule is used to compute the derivative at any node of the graph. It is the most commonly used approach in machine learning applications and neural networks, as the computational graph for the function and its derivatives can be built during the construction of the neural network, saving in future computations.  
   **Autograd** and **JAX** are the most commonly used in the frameworks to build neural networks.

Computational efficiency: automatic > symbolic > numerical

### 2. Optimization

The main application of derivatives in machine learning is optimization. You want to find the model that best fits your dataset, and derivatives help you calculate an error function that tells you how far you are from an ideal model.

When you want to optimize a function, whether maximizing or minimizing it, and the function is differentiable at every point, then it is true that the candidates for maximum and minimum are those points for which the derivative is zero. These candidates are called local minima and the absolute minimum is called the global minimum.  

![optimization0](src/optimization0.png)

#### 2.1 Optimization of Squared Loss

Square loss is a **common loss function used in regression** problems, where the goal is to predict a continuous value. In square loss, the difference between the predicted value and the actual value is squared. The objective is to minimize the average squared difference between the predicted and actual values.
![optimization0](src/optimization1.png)

#### 2.2 Optimization of Log-loss

Log loss, also known as cross-entropy loss, is commonly used as a cost function in classification problems. It is particularly useful when dealing with binary classification tasks, where the goal is to classify instances into one of two classes.  

Coin flipping game as an example:
![optimization0](src/optimization2.png)
Regular derivative method is computationally complex.
![optimization0](src/optimization3.png)
Taking the logarithm of the probability function is shown to be a simpler way to find the optimal value. 
![optimization0](src/optimization4.png)
- Maximizing the logarithm of a function is equivalent to maximizing the original function itself.
- $log(xy)=log(x)+log(y)$
- $log(x^n) = nlog(x)$
- The negative of the logarithm of the probability function, $-G(p)$, is called the log loss. This is because $G(p)$ is a negative number when p is between 0 and 1. Keeping logloss a positive value is more intuitive. 
- To maximize the original function $g(p)$, we need to minimize $-G(p)$.

![optimization0](src/optimization5.png)

### 3. Gradients - related to functions with two or more variables

#### 3.1 Tangent plane
The tangent plane is a plane that represents the slope of a function at a specific point.  
![gradient](src/gradient0.png)  
To calculate the tangent plane, you need to fix one variable and calculate the tangent line, then fix the other variable and calculate another tangent line. These two lines intersect to form the tangent plane.  
![gradient](src/gradient1.png)  

#### 3.2 Partial Derivatives
If we have a function of more than two variables, we can take partial derivatives with respect to each variable. Partial derivative with respect to X is intuitively the slope of the tangent line when treating y as a constant. 
![gradient](src/gradient2.png) 

![gradient](src/gradient3.png) 
![gradient](src/gradient4.png) 

#### 3.3 Gradient

The gradient is a **vector that contains all the partial derivatives** of a function with respect to its variables. It is denoted by the symbol nabla f and represents the slopes of the tangent plane formed by the function. 

![gradient](src/gradient5.png) 

#### 3.4 Finding Maximum and Minimum 

For a function with two or more variables, the **minimum/maximum point occurs when all partial derivatives are equal to zero**. Like how we use derivatives to find the minimum, we set all partial derivatives to zero and solve the system of equations to find the minimum or maximum points. 
![gradient](src/gradient6.png) 

![gradient](src/gradient7.png) 
Then check each candidate individually to determine if it is the minimum point.
![gradient](src/gradient8.png)  

#### 3.5 Optimization using Gradients - Analytical Method

Linear Regression with powerline example:
![gradient](src/gradient9.png) 
![gradient](src/gradient10.png)
![gradient](src/gradient11.png)

In linear regression, the cost function is a measure of how well the model fits the data. It quantifies the difference between the predicted values and the actual values of the target variable. The most commonly used cost function in linear regression is the sum of squares, also known as the mean squared error. By minimizing the sum of squares, we are finding the line that best fits the data points, as it reduces the overall error between the predicted values and the actual values. This line represents the optimal solution for the linear regression problem, and it can be used to make predictions on new data points.

### 4. Gradient Descent

Some optimization functions are hard to solve analytically.
![gradient_descent](src/gradient_descent0.png)

Instead of trying different directions, we can get smarter using a method called gradient desscent. 

#### 4.1. Optimization using gradient descent in one variable
The key idea is to **use the derivative of the function to determine the direction and magnitude of the step** towards the minimum. 
![gradient_descent](src/gradient_descent1.png)
- By subtracting the slope from the current point, we can move closer to the minimum. 
- The learning rate, denoted as alpha, determines the size of the step taken. A smaller learning rate ensures smaller steps, providing more stability and accuracy in finding the minimum. 

- The gradient descent algorithm iteratively updates the current point by subtracting the learning rate multiplied by the derivative at that point. 
- This process is repeated until the steps taken become negligible, indicating that the minimum has been reached. 

![gradient_descent](src/gradient_descent2.png)

#### 4.2 Limitations of gradient descent

1. There's no rule or definite method to find the best learning rate.

   The learning rate is a crucial parameter in machine learning that determines how quickly a model learns from the data. Finding the right learning rate can be challenging, as a rate that is too large may cause the model to miss the minimum, while a rate that is too small may result in slow convergence or failure to reach the minimum. 

2. Local Minima: Gradient descent, a commonly used optimization algorithm, may encounter the problem of getting stuck in local minima. Local minima are points that appear to be the minimum but are not the global minimum. To overcome this issue, running the gradient descent algorithm multiple times with different starting points can help find a good solution.

![gradient_descent](src/gradient_descent3.png)

#### 4.3 Optimization using gradient descent in two variables

![gradient_descent](src/gradient_descent4.png)
The algorithm starts with an initial position defined by two coordinates, $x_0$ and $y_0$. We calculate the gradient, which is the vector of partial derivatives with respect to x and y, and **move in the direction of the negative gradient**. This step is repeated until we are close enough to the minimum point. 
![gradient_descent](src/gradient_descent5.png)
