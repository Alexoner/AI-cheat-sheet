# How to ASSESS AN ALGORITHM/MODEL, such as machine learning models
Questions about off-the-shelf machine learning can be generally categorized into several classes: 

- IDEA: How does a model work? Does it make sense?
- MATHEMATICS PRINCIPAL:  Probability/statistics model, loss function, ...
- CAPABILITY: advantages vs disadvantages(pros vs cons)
- IMPLEMENTATION & MAINTENANCE: 
- RELATION: How does one model RELATE to another model

- How does one model relate to another
- Overfitting

How to assess a model?
- Performance
    - Predictive power
        - Ability to extract linear combinations of features
        - NONLINEARITY
    - Generalization
    - Robustness with respect to data distribution
        - OVERLAPPING classes
        - skewed/IMBALANCED data set
        - OUTLIER
        - MISSING data
        - mixed type of data
        - categorical/nominal data
        - deal with irrelevant inputs(data mining scenario, only small portion of inputs are relevant)
- Interpretability
- Training
    - scalability: large scale training
    - ONLINE training
    - PARALLEL training

## Data & Application

#### [How much data is enough](https://machinelearningmastery.com/much-training-data-required-machine-learning/)?
The amount of data you need depends both on the COMPLEXITY OF YOUR PROBLEM and on the COMPLEXITY OF YOUR CHOSEN ALGORITHM.

Analyze the problem, using domain expertise. Nonlinear algorithms need more data. 

Evaluate Dataset Size vs Model Skill, using [learning curve](https://en.wikipedia.org/wiki/Learning_curve).


Statistical Heuristic(rules of thumb)
- (data complexity) FACTOR OF THE NUMBER OF CLASSES: There must be x independent examples for each class, where x could be tens, hundreds, or thousands (e.g. 5, 50, 500, 5000)
- (data complexity) FACTOR OF THE NUMBER OF INPUT FEATURES: There must be x% more examples than there are input features, where x could be tens (e.g. 10).
- (model complexity) FACTOR OF THE NUMBER OF MODEL PARAMETERS: There must be x independent examples for each parameter in the model, where x could be tens (e.g. 10). 10 times as many data instances as there are features. Or analyze with VC dimension, statistical confidence interval.
 
#### deal with data having dimension(number of features) larger than number of samples
The problem is complex itself.
A: Prior/regularization

#### How to deal with unbalanced data set

Reference:
https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
https://www.kdnuggets.com/2017/06/7-techniques-handle-imbalanced-data.html

- data: sampling
- model: class weighted loss function, more appropriate model. Example: https://stats.stackexchange.com/questions/122095/does-support-vector-machine-handle-imbalanced-dataset 

How about linearly separable imbalanced data set?


#### Deal with sparse data


#### Deal with overfitting?

## Deep learning

#### Vanishing gradient problem
The cause of the problem is back when propagating through weight matrix, the same factor get multiplied multiple times, giving results tending to 0 or inifity. 
Another cause is with activation function, sigmoid activation function may kill/saturate gradients!

- Use activation function that doesn't saturate the gradient: use relu instead of tanh or sigmoid
- residual connection
- cell memory like LSTM in sequence model

#### What kind of problems does deep learning apply?
- data with large volume: thousands of parameters
- data with LOCAL CONNECTIVITY: convolution, pooling layers
- data with very high level FEATURE extraction: nonlinear activation
- data with specific data structure, like sequential structure?: recurrent neural network

#### How to understand CONVOLUTION INTEGRAL?
$$
(f*g)(t)&\,{\stackrel {\mathrm {def} }{=}}\ \int _{-\infty }^{\infty }f(\tau )g(t-\tau )\,d\tau \\&=\int _{-\infty }^{\infty }f(t-\tau )g(\tau )\,d\tau .
$$

1) Signal and system perspective
Convolution is a filtering operation, can be viewed as SLIDING WEIGHTED SUM of f, where the weights are defined by a function g. 
What's more than a weighted sum is that a function g is reflected and shifted.

What does this reflect and shift mean?
Function f(t) defines a signal, and g(t) defines a filter, the convolution integral computes the response of the (linear time invariant)system.
Filter g(t) has the meaning of effect on the signal after time t. 
Signal $f(\tau)$ at $\tau$ effected by filter $g$ for a time period of $t-\tau$ until time point $t$.
Convolution integral $(f*g)(t) = \int f(\tau)g(t-\tau)$ computes the response time t, with constantly input signal $f(\tau)$ effected by filter $g(t-\tau)$.

2) Probability perspective
Probability density function of sum of two random variables is a convolution.

Convolution can also be thought as operation of two functions f(x) and g(y), where x + y = t, which defines a hyperplane, like a paper convolving from left bottom corner.

In convolution neural networks, convolution operation is actually correlation. It doesn't matter whether to flip the operation matrix.

## Model assessment metrics
- Learning curve
For regression problem, sum of square error will do.
For classification problem, there are several metrics.
- ROC(receiver operating characteristics). ROC is diagram plotting true positive ratio against false positive ratio in response to different threshold chosen. The threshold is a value which discriminates false and negative data given the prediction score/probability.
- AUC (area under curve: ROC). Problematic for imbalanced datasets.
- F1 score: problematic imbalanced datasets

## Algorithm

### Logistic Regression
A discriminative classifier that models the target as Bernoulli distribution with probability given by a linear combination of variables.

#### Pros and cons
Performance: 
model linear combination of variables, unable to handle nonlinear effects.
Robust: the independent variables don't have to be normally distributed, or have equal variance in each group

Interpretability: easy to find the weights of each features.
Train: fast, supports online training

Cons
May underfit
Requires linearly separable.

#### Why choosing sigmoid function
Logistic regression part of the broad class of "generalized linear models", which attaches a LINK FUNCTION to the output of a linear regression model. This allows you to have non-linear fitting, but with computational tractability close to that of a linear model. For logistic regression this link function is the logit function, but you can certainly use other functions in a generalized linear model.
Interpretation of logit function is the log of odds ration:
$ logit(x) = log\dfrac{logP(y=1|x)}{logP(y=0|x)} = \vec{w}\vec{x}$

So sigmoid function is just the inverse of logit function. And it's bounded, differential, 

#### When will it not converge
1. Learning rate too large
2. Feature not normalized
3. Not linearly separable

#### Will it always get global optimum?
Loss function is convex. But the dataset may not be linearly separable, i.e., the problem may not be convex.
If the stop criteria is loss reduction under threshold, and dataset is not linearly separable, then it may not converge. Thus we can change the stop criteria to epochs trained on the dataset.

### Ensemble
 
#### Why doesn’t boosting work with linear regression?
Because of linear combination of linear regression?
The final boosting equation is equivalent to linear regression without boosting.
One observation is that a sum of subsequent linear regression models can be represented as a single regression model as well (adding all intercepts and corresponding coefficients) so I cannot imagine how that could ever improve the model. The last observation is that a linear regression (the most typical approach) is using sum of squared residuals as a loss function - the same one that GB is using.


### Decision Tree

#### pros and cons
Pros:
- easy to interpret
- robust to missing values
- often perform well on imbalanced datasets

Cons:
- Prone/sensitive to data variance because of the hierarchical structure
- doesn't support online training(not gradient descent method)
- out-of-bag(OOB) data may get the same value(falling into the same terminal node)

#### Regression tree predict using mean. How does it generalize to unseen data?

When the input data’s continuous feature value is out of bound(larger or smaller) than the train data, then the prediction value would be the same because test data are all falling into the same terminal node.


### Support Vector Machines
Structural risk minimization.

#### pros and cons

Pros
Feature mapping
Nonlinearity
Robustness

Cons
Doesn't perform well on highly skewed/imbalanced data sets
Hard to train on large scale: online, parallel
Not for online training, incremental training
Need tricks to support MULTIPLE CLASSES.

#### Why dual representation
In Lagrangian dual representation, it's easier to optimize.
Make use of kernel function in dual representation.

#### deal with multi-classes
one-versus-all, one-versus-one, train together and normalized, like softmax classifier.

#### How to output probability with SVM?
The decision boundary can be interpreted as the decision boundary of corresponding logistic regression with same weight matrix.
Then it's possible to compute the class conditional probability $P(y=1 | score) = \frac{1}{1+\exp(-score)}$.
