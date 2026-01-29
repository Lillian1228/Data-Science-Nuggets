### 1. Business Problem, Objective, and Constraints



In B2B sales scenario, our sales and client relationship team wants to drive student debt program adoptions among plan sponsors. 
- An **ROI calculation** is a key selling point for clients to adopt student debt benefit. Need to estimate how many participants have student debt. Knowing this info will greatly enhance the prospecting and sales conversations.

In B2C marketing scenario, once our plan sponsor adopt this benefit, it would be on our marketing and product teams to drive participant awareness and enrollment into this benefit.

Our goal is to create a model that can serve both scenarios: 
  1) predict SD ownership at the individual level 
  2) the predictions can be rolled up to employer level and provide accurate estimates on % of SD owners in a client.

Model has been used for both scenarios for more than 2 years and has become the default in the client 

#### Business Constrants

- **No Direct Labels**: we don't know who has student debt or not among all the parts. We identified a proxy which is a quesiton on FW survey that's available to 90% customers asking whether they are paying off SD.

- **Client-level validation**: Plan sponsors are given options to run credit bureau reports at additional cost against their employees to get the actual % of SD owners. This sets a higher bar for the model to be useful, because a large discrepancy with the client's GT could cause trust issues on the model predictions.

- **Model interpretability**: critical step to inform clients and sales team why the SD prevalence is higher than average.

### 2. Frame the Problem into an ML Task


- Dependent Variable: whether or not people are paying off their SD.
- Features: Demographic info (age, tenure, industry, gender etc.), geographics (region, % higher education), 401K activities.

Supervised learning algorithm: LightGBM
- Compared across random forest, xgboost, LightGBM
- Why: best performance and fast training.

### 4. Data Preparation

**Time Period**: Collected a dataset of 300K participants who responded they were paying off SD from 2018 to 2020. 

- From 2020 - 2023 there's a SD forgiveness program effectively pausing on SD repayment. Hence the survey responses were biased and no longer indicate SD ownership.

**Data split**: last 3 months data as holdout, remaining data 80/20 to train/validation.

**Categorical encoding**: tried one-hot encoding and target encoding on industry values. target encoding is better.

#### Resampling based on national stats




### 5. Model Training & Evaluation

#### Feature selection and hyperparameter tuning

1. Universal statistical methods: look at correlation between variables, correlation and ranking between features and labels to drop features that do not add much values.

2. Wrapper based feature selection methods

    Combine hyper parameter tuning with stepwise feature selection algorithms to drop features that do not yield a better model performance or that are deemed irrelevant:
	- Step 1 – **Bayes search for the best parameters** given the current features (50 iterations)
    - Step 2 – Run **wrapper methods (RFE, Boruta, or RFA)** on current feature set with the params found at step 1
    - Step 3 – Drop features based on step 2 output
  
  RFE chooses features following a recursive elimination procedure. RFA chooses features following a recursive addition procedure. RFA typically select features in a more conservative manner and yields lower number of variables.

  Boruta: compares features to their own randomized versions, and keeps only features with higher importance than the shuffled version. 

3. hand pick features that could be hard to deploy or in a low weight. Test these manually picked combinations and make a call to balance the model complexity and performance.

#### Probability Calibration

The client level predictions need to be probabilistically meaningful and good enough to compare with potential ground truths obtrained from credit bureau. So probability calibration is essential.

Predicted scores from LightGBM went through an isotonic calibration to achieve a better alignment with the actual fraction of positive outcomes.

#### Model Validation Methods

1. Participant level model validation
    - Validation set: 3-month hold-out sample in modeling data
    - Metrics: AUC and accuracy
    - Note: The validation shows model performance at participant level. AUC is a more reliable metric here than accuracy, because the latter depends on a choice of threshold. In final scoring, we set client specific thresholds based on probability instead of a single threshold.
    - Validation results: AUC=0.742, accuracy =0.73 (based on single threshold)

2. Client level model validation using TransUnion files
    - Validation set: 10 TransUnion files
    - Metrics: prediction deviation from TransUnion SD level
    - Note: Most relevant validation method given how the predictions will be used.
    - Validation results : +/- 4% deviation from the actual student debt % among 7 clients 

3. Client level model validation using live clients’ data
    - Validation set: 72 live clients’ loan submission data
    - Metrics: client prediction needs to be always above loan submission rate 
    - Note: A secondary validation with a larger scope on client level data points although the rule is not very rigorous.
    - Validation results : all 72 live clients have met the criteria.

4. Backtest on Participant Program Enrollment 

    Created decile reports based on model scores and calculate the enrollment rates in each decile. Among top model deciles, we saw 2 times more lifts on enrollments compared to average 

#### Fairness Testing on Gender

Gender disparity on student debt: women acount for 2/3 of student debt population.

But gender is a protected class and for bias concerns seldomly get used when building the models. In our scenario we found that including gender as a model predictor turned out to be a good way to mitigate the data bias.

Most fairness metrics we tested like equal opportunity, FNR diff, are within the acceptable range based on 80% rule. This indicates the model achieved fairly equal error rates when predicting for gender groups. The only exception is disparate impact, and  it is because it does not condition on ground truth. Breach on DI kind of makes sense and it just reflects the fact that women is more likely to have student debts than men. 



### 6. Model Inference and Serving

Monthly scoring and quarterly retraining

Client targeting dashboard: Derive client level predictions, also reporting by age/gender cuts, comparison with industry/global average, to help sales team form their talking points.


### 7. 





