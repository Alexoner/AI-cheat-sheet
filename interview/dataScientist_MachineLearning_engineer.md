# [Data Scientist / Machine Learning Engineer interview questions 

Origin: [mitbbs](https://www.mitbbs.com/article_t/DataSciences/12819.html), focusing on Amazon, Microsoft, Yelp, Pinterest, Square, Google, Glassdoor, Groupon
```text
1. Given a coin you don’t know it’s fair or unfair. Throw it 6 times and
get 1 tail and 5 head. Determine whether it’s fair or not. What’s your
confidence value?

Answer:

Reference [wikipedia](https://en.wikipedia.org/wiki/Checking_whether_a_coin_is_fair)

1) Hypothesis testing
$$
H0: the coin is fair
Ha: the coin is unfair

X is the number of heads

Rejection region: |X - 3| > 2, i.e., X = 0,1,5,or 6

significance level alpha:

alpha = P(reject H0 | H0 is true)
= P(X=0,1,5,6 | H0 is true)
= (choose(6,0)+choose(6,1)+choose(6,5)+choose(6,6))*(1/2)^6
= (1+6+6+1)*(0.5^6) = 0.21875
$$

because alpha > 0.05, we do not have enough evidence to reject H0, and we
accpte H0, so the coin is fair

confidence value?

2) Posterior probability density function of Bayesian probability theory
The posterior pdf of r(the actual probability of obtaining head in a single toss of coin),
conditional on h and t, is expressed as:
$$
f(r|H=h,T=t)={\frac {\Pr(H=h|r,N=h+t)\,g(r)}{\int _{0}^{1}\Pr(H=h|p,N=h+t)\,g(p)\,dp}}
$$
where g(r) represents the prior pdf of r, which lies in the range [0, 1].

Assuming uniform g(r) = 1, then
$$
\Pr(H=h|r,N=h+t)={N \choose h}\,r^{h}\,(1-r)^{t}
$$
Substituting this into previous formula:
$$
f(r|H=h,T=t)
={\frac {{N \choose h}\,r^{h}\,(1-r)^{t}}{\int _{0}^{1}{N \choose h}\,p^{h}\,(1-p)^{t}\,dp}}
={\frac {r^{h}\,(1-r)^{t}}{\int _{0}^{1}p^{h}\,(1-p)^{t}\,dp}}
={\frac {1}{\mathrm {B} (h+1,t+1)}}\;r^{h}\,(1-r)^{t}
={\frac {(h+t+1)!}{h!\,\,t!}}\;r^{h}\,(1-r)^{t}
$$
which is actually Beta distribution(the conjugate prior for the binomial distribution).
$$\Pr(0.45<r<0.55)=\int _{0.45}^{0.55}f(p|H=7,T=3)\,dp\approx 13\% $$


2. Given Amazon data, how to predict which users are going to be top
shoppers in this holiday season.

3. Which regression methods are you familiar? How to evaluate regression
result?

Answer:

I'm familiar with Lasso and Ridge methods.
They are both linear models, and the prediction formulation is:

$$
f(x) = beta_0 + \sum_{i=1}^p beta_i x_i

We can evaluate the regression results using mean squared error (MSE):

1/n \sum_i ( y_i - beta_0 + \sum_{i=1}^p beta_i x_i)^2

To learn the coefficients, we have

-Ridge
min \sum_i ( y_i - beta_0 + \sum_{i=1}^p beta_i x_i)^2 + lambda \sum_{i=1}^p beta_i^2

-Lasso
min \sum_i ( y_i - beta_0 + \sum_{i=1}^p beta_i x_i)^2 + lambda \sum_{i=1}^p | beta_i|
$$

4. Write down the formula for logistic regression. How to determine the
coefficients given the data?

Answer:

$$
Formula: 假设我们处理二类分类问题，y in {1,0}

Pr(y=1|x) = exp(beta' x)/(1+exp(beta' x))
Pr(y=0|x) = 1/(1+exp(beta' x))
其中beta是coefficient

y=1 if Pr(y=1|x) >= Pr(y=0|x), and y = 0, otherwise.

* Determine the coefficients given the data: 假设我们有n个data points, {(x_i,
y_i)}, i=1,..,n, where y_i in {1,0}

要通过likelihood maximization 来求beta

max_{beta} g(beta),

g(beta)是目标函数
g(beta) = sum_i log [ Pr(y=yi|x=xi)]
		= sum_i [yi beta'xi - log(1+exp(beta'xi))]

我们用Newton-Raphson update来优化这个目标函数，在每个iteration中

beta^{new} =  beta^{old} - [(g(beta)'')^-1 g(beta)']|_(beta=beta^{old})
where
g(beta)' = \sum_i xi(yi - p(yi=1|x=xi)),
g(beta)'' = - \sum_i xi xi' p(yi=1|x=xi) (1-p(yi=1|x=xi))

defining z=[y1, ..., yn]',
p = [p(yi=1|x=x1), ..., p(yi=1|x=xn)]'
W = diag(p(yi=1|x=x1)(1-p(yi=1|x=x1)), ..., p(yi=1|x=xn)(1-p(yi=1|x=xn)))
X = [x1;...;xn]

we have g(beta)' = X'(z-p), and g(beta)'' = - X' W X
$$

5. How do you evaluate regression?
For example, in this particular case:
item click-through-rate  predicted rate
1       0.04        0.06
2       0.68        0.78
3       0.27        0.19
4       0.52        0.57
...

Answer:

Using mean squared error:

$$
1/n \sum_i (click\_through\_rate_i -  predicted\_rate_i)^2

$$

6. What’s the formula for SVM? What is decision boundary?

Answer:
formula of SVM is

$$
f(x) = w'x

min_{w, xi_i} 1/2 ||w||_2^2 + C sum_i xi_i
s.t. for any i:
1 - y_i w' x_i <= xi, 0 <= xi.
$$

decision boundary:

In a statistical-classification problem with two classes, a decision
boundary or decision surface is a hypersurface that PARTITIONS the
underlying vector SPACE into two sets, one for each class. The classifier
will classify all the points on one side of the decision boundary as
belonging to one class and all those on the other side as belonging to the
other class.

x: f(x) = 0

7. A field with unknown number of rabbits. Catch 100 rabbits and put a label
on each of them. A few days later, catch 300 rabbits and found 60 with
labels. Estimate how many rabbits are there? 

Answer:

Point estimation, maximum likelihood.

100 * 300 / 60 = 500

8. Given 10 coins with 1 unfair coin and 9 fair coins. The unfair coin has
0.8532 probability to be head. Now random select 1 coin and throw it 3 times.
You observe head, head, tail. What’s the probability that the selected coin is
the unfair one?

Answer:
Bayesian rule,

$$
P(x|y) = P(x, y) / P(y) = P(y|x)P(x)/P(y)
P(y)   = \sum_{x} P(x)P(y|x).

P(unfair coin| observe head, head, tail)
= 1- P(fair coin| observe head, head, tail)

根据Bayes’ Rule：
P(fair coin| observe head, head, tail)
= P(fair coin) * P(observe | fair coin) /
 [P(fair coin) * P(observe | fair coin)+ P(unfair coin) * P(observe | unfair coin)]

其中
P(fair coin) = 9/10
P(unfair coin) = 1/10
P(observe | fair coin) = (1/2)^3
P(observe | unfair coin) = (0.8532^2)* (1-0.8532)

代入得到
P(fair coin| observe)
= (9/10*(1/2)^3)/(9/10*(1/2)^3 + 1/10*(0.8532^2)* (1-0.8532))
=  0.9132508

所以P(unfair coin| observe)
= 1 - 0.9132508
= 0.0867492
$$

9. What’s the formula for Naive Bayesian classifier? What’s the assumption
in the formula? What kind of data is Naive Bayesian good at? What is not?

Answer:


10. What is the real distribution of click-through rate of items? If you
want to build a predictor/classifier for this data, how do you do it? How do
you divide the data?

11. You have a stream of data coming in, in the format as the following:
item_id, views, clicks, time
1            100     10         2013-11-28
1            1000   350       2013-11-29
1            200     14         2013-11-30
2            127     13         2013-12-1
…

The same id are consecutive.

Click through rate = clicks / views.
On every day, I want to output the item id when its click through rate is
larger than a given threshold.
For example, at day 1, item 1’s rate is 10/100=10%, day2, its (10+350)/(100
+1000)=0.32. day3 it is (10+350+14)/(100+1000+200)=0.28.
If my threshold is 0.3, then at day 1, I don’t output. On day2 I output. On
day3, I don’t output.

11. Given a dictionary and a string. Write a function, if every word is in
the dictionary return true, otherwise return false.

12. Generate all the permutation of a string.
For example, abc, acb, cba, …

Answer: backtracking, dynamic programming, lexicographical ordering.

13. We want to add a new feature to our product. How to determine if people
like it?
A/B testing. How to do A/B testing? How many ways? pros and cons?

14. 44.3% vs 47.2% is it significant?

15. Design a function to calculate people’s interest to a place against the
distance to the place.

16. How to encourage people to write more reviews on Yelp? How to determine
who are likely to write reviews? How to increase the registration rate of
Yelp? What features to add for a better Yelp app? We are expanding to other
countries. Which country we should enter first?

Answer:
reward mechanism.

17. What’s the difference between classification and regression?

Answer: continuous versus discrete value range.

Classification tries to separate the dataset into classes belonging to the
response variable. Usually the response variable has two classes: Yes or No
(1 or 0). If the target variable can also have more than 2 categories.

Regression tries to predict numeric or continuous response variables. For
example, the predicted price of a consumer good.

The main difference between classification and regression lies on the
response they try to predict: continuous response of regression, and
discrete class label of classification.

18. Can you explain how decision tree works? How to build a decision tree
from data?

Answer: (greedy) top-down induction of decision trees, metrics(choosing
a variable and value).
Build decision trees recursively.

A decision tree has decision blocks and terminating blocks where some
conclusion has been reached. Each decision block is based on a feature/
variable/predictor. By making a decision in a decision block, we
are lead to a left/ right branch of a decision block, which is other
decision blocks or to a terminating block.

19. What is regularization in regression? Why do regularization? How to do
regularization?

Answer:
regularization is a method to improve the linear model of regression, by
shrinking the coefficients to zeros. The reason to do this is to select the
variables relevant to the response, and removing the irrelevant variables,
so that the prediction accuracy and the interpretability of the model can be
improved. The way to do regularization is to add a regularization term to
the objective of regression problem, and optimize it. This term can be a l1
norm (lasso) or l2 norm (ridge) of the coefficient vector.

20. What is gradient descent? stochastic gradient descent?

Answer:

Gradient descent is a first-order optimization algorithm. To find a local
minimum of a function using gradient descent, one takes steps proportional
to the negative of the gradient (or of the approximate gradient) of the
function at the current point. If instead one takes steps proportional to
the positive of the gradient, one approaches a local maximum of that
function; the procedure is then known as gradient ascent.

   In both gradient descent (GD) and stochastic gradient descent (SGD), you
update a set of parameters in an iterative manner to minimize an error
function.

While in GD, you have to run through ALL the samples in your training set to
do a single update for a parameter in a particular iteration, in SGD, on
the other hand, you use ONLY ONE training sample from your training set to
do the update for a parameter in a particular iteration.

Thus, if the number of training samples are large, in fact very large, then
using gradient descent may take too long because in every iteration when you
are updating the values of the parameters, you are running through the
complete training set. On the other hand, using SGD will be faster because
you use only one training sample and it starts improving itself right away
from the first sample.

SGD often converges much faster compared to GD but the error function is not
as well minimized as in the case of GD. Often in most cases, the close
approximation that you get in SGD for the parameter values are enough

21. We have a database of <product_id, name, description, price>. When user
inputs a product name, how to return results fast?

22. If user gives a budget value, how to find the most expensive product
under budget? Assume the data fits in memory. What data structure, or
algorithm you use to find the product quickly? Write the program for it.

23. Given yelp data, how to find top 10 restaurants in America?

24. Given a large file that we don't know how many lines are there. It
doesn't fit into memory. We want to sample K lines from the file uniformly.
Write a program for it.

Answer: Reservoir Sampling.

25. How to determine if one advertisement is performing better than the
other?

Answer: confidence interval, hypothesis test, control variate method?

26. How to evaluate classification result? What if the results are in
probability mode?
If I want to build a classifier, but the data is very unbalanced. I have a
few positive samples but a lot of negative samples. What should I do?

27. Given a lot of data, I want to random sample 1% of them. How to do it
efficiently?

28. When a new user signs up Pinterest, we want to know its interests. We
decide to show the user a few pins, 2 pins at a time. Let the user choose
which pin he/she likes. After the user clicks on one of the 2, we select
another 2 pins.
Question: how to design the system and select the pins so that we can
achieve our goal?

Answer: binary search, similar metric in the decision tree learning.
At each step, we choose the criterion which can maximize the information
gain(reduce the entropy).

29. Write a function to compute sqrt(X). Write a function to compute pow(x,
n) [square root and power)

Answer: 1) binary search 2) Newton's method

30. Given a matrix
a b c d
e f g h
i j k l

Print it in this order:

a f k
b g l
c h
d
e j
i

31. Given a matrix and an array of words, find if the words are in the
matrix. You can search the

matrix in all directions:  from left to right, right to left, up to down,
down to up, or diagonally.
For example
w o r x b
h e l o v
i n d e m

then the word “world” is in the matrix.

Answer: Graph search(bfs/dfs).


32. Given a coordinates, and two points A and B. How many ways to go from A
to B? You can only move up or right.
For example, from (1, 1) to (5, 7), one possible way is 1,1 -> 2, 1… 5, 1 -
> 5,2 -> ..5, 7

Answer: Combinatorial number.


33. In a city where there are only vertical and horizontal streets. There
are people on the cross point. These people want to meet. Please find a
cross point to minimize the cost for all the people to move.

34. Design a job search ranking algorithm on glassdoor

35. How to identify review spam?

36. Glassdoor has this kind of data about a job : (position, company,
location, salary). For example (Software Engineer, Microsoft, Seattle, \$125K
). For some records, all four entires are available. But for others, the
salary is missing. Design a way to estimate salary for those records.

37. When to send emails to users in a day can get maximum click through rate?

38. Youtube has video play log like this:
Video ID, time
vid1        t1
vid2        t2
...           ...
The log is super large.
Find out the top 10 played videos on youtube in a given week.

39. Write a program to copy a graph

40. A bank has this access log:
IP address, time
ip1      t1
ip2      t2
...        ...

If one ip accessed K times within m seconds, it may be an attack.
Given the log, identify all IPs that may cause attack.


1. 问 Skill Set 以及对于常见工具的掌握。
　　Skill Set 就是指你掌握了哪些知识，一般问起来都是比较粗略地问，主要目的就是考察和团队的习惯
以及工具的掌握是否 Match。各种基础问题，比如计算机网络中 HTTP、TCP、UDP 协议，数据库的设计原
则、实现方法，操作系统的一些基本知识，Unix 的常见指令，Hadoop 和 Hadoop Streaming 如何使用、
如何 Debug ，平时使用什么 IDE 什么 OS。

2. 问简历，就简历上的技术细节发问，主要是项目有关的技术细节，以及相关的技术延伸。
　　比如项目中就提到了 NLP 相关的东西，就问一些和 NLP 相关工具的使用，比如 Stanford NLP 等。
再又问了一些延伸的问题，比如，如何自动生成一个有意义的句子，如何把一段文字 Split 成一个个句子，
怎么选 feature 怎么做 model 等等。这类问题主要还是需要对于自己的项目技术细节足够了解，且对于延伸的问题有所掌握。

3. Machine Learning、Statistic 的相关问题

一些分布参数的最大似然估计之类的东西是什么，如何推导
LR SVM 的本质区别是什么
哪些 Regularization，都各有什么性质
对于 Naive Bayes 的理解，NB 有哪些局限性
Random Forest 为什么很好用
如何做 Model Selection
给一组数据，问 Decision Tree，LR，NB，SVM 等算法学出来都是什么样子的，是否学不出来，怎么处理，有哪些 Kernel，在图上画线怎么画
一些比较难的问题，比如：

对于 Graphical Model 的理解，写出 LDA 的公式，给出 Topic Model 生成过程等的
PageRank 的原理和公式推导

4. 给一个现实问题，如何解决。

　　这一类问题就比较宽泛了，主要是思考的框架。比如如何收集数据，收集那些数据，如何定 feature，如何定 measurement，如何定 milestone 等等。要分层次一步一步地讨论。

　　举个例子，比如要你做一个房地产的搜索引擎，该怎么做？

　　最后，感觉很多东西还是得从做项目中来学习。所以还在读书的同学还是得想办法多做一些实际的项目，最好是有真实世界数据的，这样就可以经历一些 Clean Data 等耗时耗力，老师不教但是在实际工作中又非常有用的过程，帮助自己成长。同时，还是要尽量地把一个项目的时间做的长一些，比如 6 个月，8 个月，才有可能出比较理想的成果。


==============================================================================================
==============================================================================================

2. While it is often assumed that the probabilities of having a boy or a girl are the same, the actual probability of having a boy is slightly higher at 0.51. Suppose a couple plans to have 3 children. What is the probability that exactly 2 of them will be boys?

A. 0.38, ✓
B. 0.48
C. 0.78
D. 0.68
E. 0.58
F. I'm not sure.

* 3. Given a Gaussian distribution with mean of 2 and standard deviation = 4, what is the cumulative probability at 2?

A. 0.03

B. 0.5, ✓

C. 0.023

D. 0.25

E. I'm not sure.

* 4. About 30% of human twins are identical, and the rest are fraternal. Identical twins are necessarily the same sex, half are males and the other half are females. One-quarter of fraternal twins are both male, one-quarter both female, and one-half are mixes: one male, one female. You have just become a parent of twins and are told they are both girls. Given this information, what is the probability that they are identical?

A. 33

B. 46, ✓

C. 72

D. 50

E. I'm not sure.

* 5. A New York City cab was involved in a hit-and-run accident last night. Five witnesses reported the incident, four of whom said that the cab was green and one of whom said that the cab was yellow. Assume each witness correctly identifies the color of a cab with probability 2/3. It is known that 85% of registered cabs in New York City are yellow and 15% are green. Based on this information, what is the probability that the cab was green?

A. 58.5%, ✓

B. 88.9%

C. 85%

D. 66.6%

E. I'm not sure.

Answer:

We want to compute the POSTERIOR PROBABILITY or CONDITIONAL PROBABILITY that the car was
green (G), given the witnesses, P(G|w).

From BAYES THEOREM, we need to compute the conditional probability
	P(G|w) = \frac{P(w|G)P(G)}{P(w)}

We know PRIOR PROBABILITY P(G)=0.15 and P(Y)=0.85 (Y stands for yellow).

Lets compute P(w). By definition,
	P(w) = \sum_{c\in \{G,Y\}} P(w,c) \\
		 = \sum_{c\in \{G,Y\}} P(w|c)P(c) \\
		 = P(w|G)P(G) + P(w|Y)P(Y).

P(w|c) is the LIKELIHOOD probability of the particular witnesses outcome given the color of
the car, which obeys the BINOMIAL DISTRIBUTION,
	P(w|G)=5×(2/3)⁴ * 1/3
	P(w|Y)=5×(1/3)⁴ * 2/3

Plugging in in the equation of P(w)P(w), and in the equation for P(G|w)P(G|w) in the
numerator, and doing the math:
P(G|w) = [0.15 * 5×(2/3)⁴ * 1/3] / [0.15 * 5×(2/3)⁴ * 1/3 + 0.85×5×(1/3)⁴ * 2/3]
	   ≈ 0.585


* 6. Let f be the function defined by f(x) = 2x3  - 6x2 + 5x - 5
Is this function increasing or decreasing at x=1?

A. Increasing

B. Decreasing, ✓

C. Neither Increasing nor Decreasing

D. I'm not sure.

==============================================================================================
==============================================================================================

Field problems

1. A jar has 1000 coins, of which 999 are fair and 1 is double headed. Pick a
coin at random, and toss it 10 times. Given that you see 10 heads, what is the
probability that the next toss of that coin is also a head? ([origin](https://news.ycombinator.com/item?id=6999884))

Answer:
Hints: Bayesian, conditional probability, turn a problem into mathematics.

Define variables/events:
o = observation of trial,
f = the selected coin fairness(fair or unfair),
t = the toss of the coin (head or tail)

And the value O = Observation = 10 heads.

Given prior probability p(f): p(f=fair) = 0.999, p(f=unfair) = 0.001.

Now, compute the likelihood probability for observation by marginalization of
conditional probability.
$$
p(o) = \sum_f p(o, f)
p(o, f) = p(f) * p(o | f)
$$

p(O|f = fair) = (1/2) ^ 10, p(O|f = unfair) = 1.
So,
$$
p(O) = p(o = O) = \sum_f p(O, f) = 0.999 * (1/2)^{10}  + 0.001 * 1
$$

Then, we can compute the conditional posterior probability p(f|o).
$$
p(y|x) = p(x, y) / p(x) = p(x, y) / \sum_y p(x, y)
       = p(y) * p(x | y) / \sum_y p(y) * p(x | y)
$$
Substituting x, y with o and f, we have:
$$
p(fair | O) = p(fair) / p(O)
p(unfair | O) = p(unfair) / p(O)
$$

Then the conditional probability with multiple variables' Bayes rule :
$$
p(t, f | o) = p(t | f, o) p(f | o)
p(t | o) = \sum_f p(t, f | o) = \sum_f p(f | o)p(t | f, o)
         = \sum_f p(f | o) p(t | f)
$$
So,
p(head | O) = p(fair | O) * 0.5 + p(unfair | O) * 1
            = (999 / 2 + 1024) / (999 + 1024) = 0.7530894710825506

2. Given a coin with unknown probability of flipping heads, toss the coin and get only
1,000,000 heads, then what's the probability of flipping head of next toss.

Answer: Bayesian inference

Denote the trial results as $$X_i$$, and the probability of toss head as $$\theta$$, then the
posterior predictive distribution is:
$$
p(x_{n+1} = 1) = \int p(x_{n+1} = 1|\theta) p(\theta | X)
$$

The observation is i.i.d CONDITIONED on the fairness p of coin, and obey binomial distribution.
We can model the predictive distribution of next toss flip as POSTERIOR distribution over
observation.

Take Beta distribution Beta(a, b), as PRIOR DISTRIBUTION, then it's a CONJUGATE PRIOR. So the
posterior distribution is Beta(a + k, b + n - k).

To make point estimation, we compute the integral of posterior, and result is the mean
$$\theta = (a + k) / (a + b + n)$$.

B.t.w, we can use Z-score and so on to do hypothesis testing.
If we choose other prior distributions, we might need to adopt MCMC to calculate the integral.

```
