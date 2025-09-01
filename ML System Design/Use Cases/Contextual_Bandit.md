





### Model Development

- Data Challenge

    The reward signal in this domain is very sparse because users usually do not click on an engager, and user clicking is very random so that returns have high variance.

- Methods

    1. Greedy optimization: maximizing only the immediate clicks and did not take the long-term effects of recommendations into account.

        Use a supervised model to estimate the probability of a click as a function of user features. Then we define an "epsilon-greedy policy that selected with probability 1-epsilon the engager predicted by the supervised model to have the highest probability of producing a click, and otherwise selected from the other engagers uniformly at random.



### Findings

#### Limitation

- Short-term greedy policy and long-term trade-off

    Policies derived from the contextual bandit formulation are greedy in the sense that they do not take long-term effects of actions into account. These policies effectively **treat each visit to a website as if it were made by a new visitor uniformly sampled from the population of the website’s visitors**. By **not using the fact that many users repeatedly visit the same websites**, greedy policies do not take advantage of possibilities provided by long-term interactions with individual users.