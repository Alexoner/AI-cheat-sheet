# Aspects of machine learning models
Questions about machine learning can be generally categorized into several classes: 

- How does a model work? What's the IDEA of a model?
- What are the ADVANTAGES AND DRAWBACKS of a model?
- What's the MATHEMATICS behind a model?
- How does one model RELATE to another model

- The mathematical model
- loss function
- advantages vs disadvantages(pros vs cons)
- How does one model relate to another
- Overfitting

## Data
Q:How much data is enough?
A: A rule of thumb: 10 times as many data instances as there are features.
Or analyze with VC dimension, statistical confidence interval.
 
Q: data dimension(number of features) larger than number of samples
A: Prior/regularization
 
Q: Why doesn’t boosting work with linear regression?
Because of linear combination of linear regression?
The final boosting equation is equivalent to linear regression without boosting.
One observation is that a sum of subsequent linear regression models can be represented as a single regression model as well (adding all intercepts and corresponding coefficients) so I cannot imagine how that could ever improve the model. The last observation is that a linear regression (the most typical approach) is using sum of squared residuals as a loss function - the same one that GB is using.

Q: Regression tree predict using mean. How does it generalize to unseen data?

When the input data’s continuous feature value is out of bound(larger or smaller) than the train data, then the prediction value would be the same because test data are all falling into the same terminal node.



## Decision Tree

### pros and cons
Pros:
- easy to interpret
- robust to missing values
-

Cons:
- Prone/sensitive to data variance because of the hierarchical structure
- doesn't support online training(not gradient descent method)
- out-of-bag(OOB) data may get the same value(falling into the same terminal node)
