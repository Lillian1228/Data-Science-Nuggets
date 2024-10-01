### 1. Intro to Probability

#### 1.1 Concept

- Probability is a measure of how likely an event is to occur. In mathematics, we use probability to quantify uncertainty. By calculating probabilities, we can make informed decisions and understand the likelihood of different outcomes.  

- The complement of an event is basically the probability that the event does not happen: $P(not A) = 1 - P(A)$
- Sum of probabilities (disjoint events): P(A OR B) = P(A) +P(B)
- Sum of probabilities (joint events): P(A OR B) = P(A) + P(B) - P(AB)

![NM](src/prob1.png)

- Independence: the occurrence of one event does not affect the probability of another event happening. 
  - product rule for independent events: $P(AB) = P(A)*P(B)$

- Conditional Probability:  the probability of an event happening, given that another event has already occurred. It helps us see how new information can change the probability of an event. It's like adjusting our expectations based on what we already know.
  - General product rule: $P(AB) = P(A)*P(B|A)$, where P(B|A)=P(B) when independent.
  - $P(B|A) = P(AB) / P(A)$

#### 1.2 Bayes Theorem

- Intuition  
Bayes' theorem is a fundamental concept in probability and is widely used in various fields, including machine learning. It helps calculate the probability of an event occurring given certain conditions. In the context of a medical test, Bayes' theorem can be used to determine the probability of having a disease given a positive test result. 

![NM](src/prob2.png)

![NM](src/prob3.png)

![NM](src/prob4.png)

If a person receives a positive test result, the probability of actually having the disease is influenced by the base rate.  The base rate of a condition refers to the prevalence or likelihood of that condition in a population. When interpreting test results, the base rate plays a crucial role in determining the probability of having the condition given a positive or negative test result.

- Formula
![NM](src/prob5.png)

- Prior and Posterior

The **prior** probability is the initial probability, the event provides new information, and the **posterior** probability is the updated and more accurate probability based on that information.

- Naive Bayes Model
  
The naïve Bayes algorithm is a machine learning method used for classification tasks, such as determining if an email is spam or not.

It calculates the probability of a certain class (spam or ham) given the presence of specific words in the document.

![NM](src/prob6.png)
The algorithm assumes that the appearance of each word is independent of the others, even though this is not always true in reality.
![NM](src/prob7.png)
It calculates the prior probabilities of each class (spam or ham) based on the frequency of those classes in the dataset.

The algorithm also calculates the conditional probabilities of each word given each class.

To classify a new document, it multiplies the prior probability of each class by the conditional probabilities of the words in the document given each class.
![NM](src/prob8.png)
The class with the highest probability is assigned as the predicted class for the document.

Despite the simplifying assumption, the naïve Bayes algorithm can still provide useful results in many cases.

- ML Applications

![NM](src/prob9.png)

![NM](src/prob10.png)

### 2. Probability Distributions

 
- Deterministic variables are associated with a fixed outcome, and random variables with an uncertain outcome. 
  ![NM](src/dist1.png) 
- The **probability mass function (PMF)** is a function that describes the probability distribution of a discrete random variable. It assigns probabilities to each possible outcome of the random variable. 
![NM](src/dist2.png) 

#### 2.1 Bernoulli Distribution

- Bernoulli distribution is a probability distribution that represents a random experiment with two possible outcomes - success and failure. 
- Random variable (denoted as x): In the context of the Bernoulli distribution, we use a random variable to represent the number of successes in the experiment.
- Probability of success (denoted as p): the likelihood of obtaining a success in the experiment. 
- Parameter of the Bernoulli distribution: only one is the probability of success (p). This parameter determines the shape and characteristics of the distribution.
- The probability mass function (PMF) of the Bernoulli distribution is given by:

  $P(X = x) = p^x * (1 - p)^(1 - x)$, where:
  - X is the random variable representing the outcome of the experiment (0 for failure, 1 for success).
  - p is the probability of success.
- The mean (expected value) of the Bernoulli distribution is given by: E(X) = p
- The variance of the Bernoulli distribution is given by: Var(X) = p * (1 - p)

#### 2.2 Binomial Distribution

- The binomial distribution is a discrete probability distribution that models a series of independent Bernoulli trials, where each trial has two possible outcomes - success and failure. It is defined by two parameters: the number of trials, n, and the probability of success in each trial, p.
- The probability mass function (PMF) of the binomial distribution is given by:

  $P(X = k) = C(n, k)  p^k  (1 - p)^(n - k)$, where:
  - X is the random variable representing the number of successes in the n trials.
  - k is the specific number of successes.
  - n is the total number of trials.
  - p is the probability of success in each trial.
  - C(n, k) is the binomial coefficient, which represents the number of ways to choose k successes from n trials and is calculated as C(n, k) = n! / (k! * (n - k)!), where ! denotes the factorial function.  
- The mean (expected value) of the binomial distribution is given by: E(X) = n * p
- The variance of the binomial distribution is given by: Var(X) = n  p  (1 - p)
- The **binomial coefficient**, denoted as "n choose k" or "C(n, k)", is a mathematical term that represents the number of ways to choose k items from a set of n items, without regard to the order of selection: $C(n, k) = n! / (k! * (n - k)!)$, where "!" denotes the factorial function. 
![NM](src/dist3.png) 
- The binomial coefficient has several properties, such as symmetry ($C(n, k) = C(n, n - k)$) and the fact that it represents the coefficients in the expansion of a binomial expression, such as (a + b)^n. 
- In the context of the binomial distribution, the binomial coefficient is used to calculate the number of ways to obtain a specific number of successes (k) in a fixed number of trials (n).  
![NM](src/dist4.png)  
![NM](src/dist5.png)  
![NM](src/dist6.png)  
![NM](src/dist7.png)  

#### 2.3 Continuous Probability Distributions

Discrete distributions involve events that can be listed and have specific values, while continuous distributions involve events that cannot be listed and have a range of values within an interval.

Discrete distributions:
- Events can be listed and take on specific values.
- Examples include the number of times a coin is tossed, the number of people in a town, or the number of cars passing by in an hour.
- The probabilities of each possible value can be represented as bars in a histogram or a probability mass function.
- The sum of the probabilities of all possible values is equal to 1.

Continuous distributions:
- Events cannot be listed and take on a range of values within an interval.
- Examples include the amount of time spent waiting on the phone, the height of individuals in a population, or the temperature in a given location.
- The probabilities are represented as a curve, often called a probability density function.
- The area under the curve represents the probability, and the height of the curve at a specific point does not directly represent the probability of that exact value.
- The probability of a specific value in a continuous distribution is typically zero, as there are infinitely many possible values within any given interval.
  ![NM](src/dist8.png)  

Probability Density Function (PDF): The PDF is a function that represents the rate at which probability accumulates around each point in a continuous distribution. It is denoted as lowercase f and is used to calculate probabilities by finding the area under the PDF curve between two points.

  ![NM](src/dist9.png)  

Cumulative Distribution Function (CDF): 
- The CDF shows the probability that a random variable is less than or equal to a certain value. It is a more convenient way to calculate probabilities compared to finding areas under a curve.
- The CDF starts at 0 and ends at 1, always positive and increasing, representing the entire range of probabilities.
- For discrete distributions, the CDF has jumps at the possible outcomes of the random variable. The height of each jump represents the probability of that specific outcome.
- For continuous distributions, the CDF is a smooth curve without jumps. The CDF is obtained by adding up the areas under the PDF curve from the beginning to a specific value.

  ![NM](src/dist10.png) 

  #### 2.4 Uniform Distribution

  - In uniform distribution, all values within a given interval have the same probability of occurring. 
  - The PDF of a uniform distribution is constant within the interval and 0 outside the interval. The height of the PDF is determined by the length of the interval.  
  - The CDF of a uniform distribution is a straight line with a slope of 1 within the interval. It starts at 0 and ends at 1.  
  - The uniform distribution has two parameters, a and b, which represent the beginning and end of the interval. The PDF is 1/(b-a) within the interval.
![NM](src/dist11.png) 
![NM](src/dist12.png) 

#### 2.5 Normal/Guassian Distribution

Intuition: When the number of trails n in binomial distribution is very large, the binomial distribution can be approximated with guassian distribution pretty well.

![NM](src/dist13.png) 

- The formula for the normal distribution includes parameters such as the mean (mu) and standard deviation (sigma), which determine the center and spread of the data. 
- The standard normal distribution is a special case with a mean of 0 and a standard deviation of 1. 
![NM](src/dist14.png) 
![NM](src/dist15.png) 
![NM](src/dist16.png) 
![NM](src/dist17.png) 

#### 2.6 Chi-Squared Distribution


![NM](src/dist18.png)  
- CDF of Chi-sqaure distribution with one degree of freedom can be obtained from PDF of the standard normal distribution.
![NM](src/dist18-1.png)  
- PDF is the derivative of the CDF  
![NM](src/dist18-2.png)
- The distribution of the accumulated noise power over multiple transmissions is described using the Chi-squared distribution with different degrees of freedom. 
The distribution becomes more spread and symmetrical as the number of transmissions k increases.
![NM](src/dist19.png) 

#### Sampling from a distribution

1. Sampling from CDF of a discrete distribution
  ![NM](src/dist20.png) 

2. Sampling from CDF of a continuous distribution
    ![NM](src/dist21.png) 

