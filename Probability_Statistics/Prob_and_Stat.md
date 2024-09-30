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
