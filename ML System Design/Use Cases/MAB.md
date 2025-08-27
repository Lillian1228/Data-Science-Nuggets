### 1. Business Problem

The aim of ‘Next Best Action’ is to provide or recommend to participants who visit the NetBenefits website a set of actions (ranked in order of likelihood to get clicked) that will enable them to plan a life they envision and take the next steps to execute on that plan. Currently, there are 4 methods or spinpaths to show the actions to the participants: 
- Random selection of the action cards
- Financial wellness heuristics
- User similarity model
- Micro-segmentation of participants model

These 4 methods are allocated to participants randomly but manually at regular intervals and the impact or click-through-rates of the methods are assessed through experiments. **Based on the performance of the spinpaths, the percentages are reallocated manually and the process repeats**. 

### 2. Business Objective

To reduce manual intervention, we require a process that will **automatically modify the percentage allocations as and when the spinpaths outperform each other**. This is to identify the best Spinpath for a particular timeframe, but also identify other spinpaths that could potentially be the best at a future time.

### 3. Business Constraints / Requirements

- Evolving Catalog 

### 4. Frame the Problem as ML Task

- ML Objective

Maximize the overall rewards (clicks) by optimizing the traffic allocation among all the spinpaths.

- System Inputs
    - Users and their snapshot features
    - Item features: embeded card titles
    - User-Item Interactions: impressions, clicks
    - Context features: treatment groups (arms), click day and months

- System Outputs

    - 


- diagram



### 5. Choosing the right method: Why MAB?





### 6. Feature Engineering



### 7. Model Development




### 8. Offline and Online Evaluation




### 9. Deployment and Serving

