# Data Scientist / Machine Learning Engineer interview questions
## from http://leetcode0.blogspot.jp/2014/12/data-scientist-machine-learning-engineer.html (Amazon, Microsoft, Yelp, Pinterest, Square, Google, Glassdoor, Groupon)

1. Given a coin you don’t know it’s fair or unfair. Throw it 6 times and 
get 1 tail and 5 head. Determine whether it’s fair or not. What’s your 
confidence value? 

2. Given Amazon data, how to predict which users are going to be top 
shoppers in this holiday season. 

3. Which regression methods are you familiar? How to evaluate regression 
result? 

4. Write down the formula for logistic regression. How to determine the 
coefficients given the data? 

5. How do you evaluate regression? 
For example, in this particular case:
item click-through-rate  predicted rate
1       0.04        0.06
2       0.68        0.78
3       0.27        0.19
4       0.52        0.57
…

6. What’s the formula for SVM? What is decision boundary? 

7. A field with unknown number of rabbits. Catch 100 rabbits and put a label
on each of them. A few days later, catch 300 rabbits and found 60 with 
labels. Estimate how many rabbits are there?  

8. Given 10 coins with 1 unfair coin and 9 fair coins. The unfair coin has &
#8532; prob. to be head. Now random select 1 coin and throw it 3 times. You 
observe head, head, tail. What’s the probability that the selected coin is 
the unfair one? 

9. What’s the formula for Naive Bayesian classifier? What’s the assumption
in the formula? What kind of data is Naive Bayesian good at? What is not? 

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

17. What’s the difference between classification and regression? 

18. Can you explain how decision tree works? How to build a decision tree 
from data? 

19. What is regularization in regression? Why do regularization? How to do 
regularization? 

20. What is gradient descent? stochastic gradient descent?

21. We have a database of <product_id, name, description, price>. When user 
inputs a product name, how to return results fast? 

22. If user gives a budget value, how to find the most expensive product 
under budget? Assume the data fits in memory. What data structure, or 
algorithm you use to find the product quickly? Write the program for it. 

23. Given yelp data, how to find top 10 restaurants in America?

24. Given a large file that we don’t know how many lines are there. It 
doesn’t fit into memory. We want to sample K lines from the file uniformly.
Write a program for it. 

25. How to determine if one advertisement is performing better than the 
other? 

26. How to evaluate classification result? What if the results are in 
probability mode? 
If I want to build a classifier, but the data is very unbalanced. I have a 
few positive samples but a lot of negative samples. What should I do?

27. Given a lot of data, I want to random sample 1% of them. How to do it 
efficiently? 

28. When a new user signs up Pinterest, we want to know its interests. We 
decide to show the user a few pins, 2 pins at a time. Let the user choose 
which pin s/he likes. After the user clicks on one of the 2, we select 
another 2 pins. 
Question: how to design the system and select the pins so that we can 
achieve our goal? 

29. Write a function to compute sqrt(X). Write a function to compute pow(x, 
n) [square root and power)


30. Given a matrix
a b c  d
e f  g  h
i  j  k   l
Print it in this order:
a  f  k
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
h  e l  o v
i   n d e m

then the word “world” is in the matrix. 


32. Given a coordinates, and two points A and B. How many ways to go from A 
to B? You can only move up or right. 
For example, from (1, 1) to (5, 7), one possible way is 1,1 -> 2, 1… 5, 1 -
> 5,2 -> ..5, 7


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


## CMU-CS master 北美数据科学

陈然，THU 软件学院 2009 级，CMU-MCDS 13Fall，暑假在 MCDS Director Prof. Eric Nyberg 的 OAQA 组里干活，一直觉得很有可能留下来继续读 PhD 的，做做 Machine Learning 的交叉学科的应用。后来被老板告知没有 RA（毕业至 15Fall PhD 入学这段时间），要去工作，于 9 月的 Career Fair 开始刷题找工作，因为准备得晚，一直被拒，连跪七个面试后来了一个大 offer，Data Scientist @ Trulia in SF ，考虑到这个组很小，只有 5 个人，也是偏 Research 的，其他人都是 PhD 或者有多年工作经验的，再加上估计也不会有人给我更多的钱了，遂从了，并毫不犹豫地放弃了 PhD 的学术理想。
面试的职位包括：Data Scientist，Data Engineer，Software Engineer in Machine Learning，Data Analyst 等。当然其中有不少也包括最常见得 Leetcode Style 的算法题，除了这一类题目以外，还有不少其他类型的题目，主要分为这么几类：

1. 问 Skill Set 以及对于常见工具的掌握。

　　Skill Set 就是指你掌握了哪些知识，一般问起来都是比较粗略地问，主要目的就是考察和团队的习惯以及工具的掌握是否 Match。我被问到过各种各要的碎碎的问题，比如计算机网络中 HTTP、TCP、UDP 协议，数据库的设计原则、实现方法，操作系统的一些基本知识，Unix 的常见指令，Hadoop 和 Hadoop Streaming 如何使用、如何 Debug，平时使用什么 IDE 什么 OS……总之各个琐碎的角落都被问到过。

2. 问简历，就简历上的技术细节发问，主要是项目有关的技术细节，以及相关的技术延伸。

　　比如我的项目中就提到了 NLP 相关的东西，就被问了一些和 NLP 相关工具的使用，比如 Stanford NLP 等。再又问了一些延伸的问题，比如，如何自动生成一个有意义的句子，如何把一段文字 Split 成一个个句子，怎么选 feature 怎么做 model 等等。这类问题主要还是需要对于自己的项目技术细节足够了解，且对于延伸的问题有所掌握。

3. Machine Learning、Statistic 的相关问题

　　Machine Learning 相关的问题就太多了，我稍微列举一些我遇到过的问题：

一些分布参数的最大似然估计之类的东西是什么，如何推导
LR SVM 的本质区别是什么
哪些 Regularization，都各有什么性质
对于 Naive Bayes 的理解，NB 有哪些局限性
Random Forest 为什么很好用
如何做 Model Selection
给一组数据，问 Decision Tree，LR，NB，SVM 等算法学出来都是什么样子的，是否学不出来，怎么处理，有哪些 Kernel，在图上画线怎么画
　　还有被问到了一些比较难的问题，比如：

对于 Graphical Model 的理解，写出 LDA 的公式，给出 Topic Model 生成过程等的
PageRank 的原理和公式推导
　　总之，前面那些问题本质上都不是那么难，但是不少问题都需要对于 ML 各种知识的融会贯通，所以大家在学习的时候还是需要深入学习，不要浮于表面。

4. 给一个现实问题，如何解决。

　　这一类问题就比较宽泛了，主要是在回答的时候记住考察的目的很多时候并不是技术本身，而是你对于这一类问题没有思考的框架。比如如何收集数据，收集那些数据，如何定 feature，如何定 measurement，如何定 milestone 等等。要分层次一步一步地讨论。

　　举个例子，比如要你做一个房地产的搜索引擎，该怎么做？

　　最后，感觉很多东西还是得从做项目中来学习。所以还在读书的同学还是得想办法多做一些实际的项目，最好是有真实世界数据的，这样就可以经历一些 Clean Data 等耗时耗力，老师不教但是在实际工作中又非常有用的过程，帮助自己成长。同时，还是要尽量地把一个项目的时间做的长一些，比如 6 个月，8 个月，才有可能出比较理想的成果。
