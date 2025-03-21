#### Fundamental problem of causal inference

We can never observe the same unit with and without treatment. We call the potential outcome that happened *factual* and the one that didn't happen *counterfactual*.

Notation:

- $Y_i(0)$ is the potential outcome for unit i without the treatment. 
- $Y_i(1)$ is the potential outcome for unit i with the treatment.

1. **Individual treatment effect (ITE)**: measures the effect of a treatment on a specific individual. 
   
   $ITE_i = Y_i(1) - Y_i(0)$

   However, we can never observe the individual treatment effect (fundamental problem of causal inference).

2. **Average Treatment Effect**: average treatment effect across all individuals. Given the fundamental problem of causal inference, it is typically estimated using techniques like RCT, matching, regression, inverse probability weighting, etc.

    $ATE = E[Y_1 - Y_0]$