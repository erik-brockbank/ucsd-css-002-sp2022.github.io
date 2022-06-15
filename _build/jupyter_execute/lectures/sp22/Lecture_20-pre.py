#!/usr/bin/env python
# coding: utf-8

# # Lecture 20 (5/11/2022)

# **Announcements**
# 
# 

# *Last time we covered:*
# - ROC curves
# 
# **Today's agenda:**
# - Common classification models
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split


# # Common Classification Models
# 
# - $k$-nearest neighbors
# - Logistic regression
# - Decision trees
# - Support Vector Machines (SVMs)
# - Other: naive Bayes, neural networks, discriminant analysis
# 

# ## Data: Predicting Heart Disease
# 
# 
# From [source](https://hastie.su.domains/ElemStatLearn/):
# 
# > A retrospective sample of males in a heart-disease high-risk region
# of the Western Cape, South Africa. There are roughly two controls per
# case of CHD. Many of the CHD positive men have undergone blood
# pressure reduction treatment and other programs to reduce their risk
# factors after their CHD event. In some cases the measurements were
# made after these treatments. These data are taken from a larger
# dataset, described in  Rousseauw et al, 1983, South African Medical
# Journal. 
# 
# - sbp: systolic blood pressure
# - tobacco: cumulative tobacco (kg)
# - ldl: low densiity lipoprotein cholesterol
# - adiposity
# - famhist: family history of heart disease (Present, Absent)
# - typea: type-A behavior
# - obesity
# - alcohol: current alcohol consumption
# - age: age at onset
# - chd: **response**, coronary heart disease

# In[2]:


data = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data')

data


# **Setting up our classifiers**:
# 
# Let's stick to just a single feature (age at onset) and see how different methods use this feature to predict the outcome label (CHD). 

# In[3]:


x_vals = np.array(data['age']).reshape(len(data), 1)
y_vals = np.array(data['chd'])

xtrain, xtest, ytrain, ytest = train_test_split(x_vals, y_vals, random_state = 1)


# ***Now, let's get started!***

# ## Logistic Regression
# 
# **How it works**:
# 
# In linear regression, the relationship between our predictor $x$ and our response variable $y$ was:
# 
# $y = \beta_0 + \beta_1 x$
# 
# If our $y$ values are all 0 or 1 (and assumed to be *Bernoulli distributed* with probability $p$), this approach doesn't work very well:
# 1. It predicts values <0 and >1 for some inputs $x$
# 2. It doesn't accomodate the fact that getting closer and closer to 1 gets harder and harder: one-unit changes in $x$ may not have equal changes in $p(y = 1)$. 
# 
# *So what do we do about this?*
# 
# Instead, we postulate the following relationship between $x$ and $y$:
# 
# $log \dfrac{p(y=1)}{p(y=0)} = \beta_0 + \beta_1 x$.
# 
# Every unit increase in $x$ leads to a $\beta_1$ increase in the *log odds of $y$* (or, every unit increase in $x$ leads to a $\beta_1$ *multiplication* of the *odds* of $y$).
# 
# This *logit transform* of our response variable $y$ solves both of the problems with linear regression above. 
# 
# However, the goal today isn't to get into the nitty-gritty of logistic regression. Instead, let's talk about how we use it as a classifier!
# 
# **Classification**
# 
# When we've fit a logistic regression to our data, we can output a probability $p(y)$ for any given $x$:
# 
# $p(y) = \dfrac{e^{h(x)}}{1+ e^{h(x)}}$
# 
# for $h(x) = \beta_0 + \beta_1x$.
# 
# $\dfrac{e^{h(x)}}{1+ e^{h(x)}}$ is the *logistic function* that maps from our $x$ variable to $p(y)$.
# 
# We can use this function as the basis for classification, where $p(y)$ greater than a threshold $T$ is given a particular label estimate $\hat{y}$.  
# 
# 
# **Fitting parameters**
# 
# Even though logistic regression produces regression coefficients (intercept + slopes) similar to linear regression, these parameters are not estimated using the Ordinary Least Squares process we saw with linear regression. Instead, they are most often estimated using a more complicated process called Maximum Likelihood Estimation. 
# 

# ### Logistic regression in python
# 
# You can read the scikit-learn documentation [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
# 

# In[4]:


# Import the LogisticRegression class
from sklearn.linear_model import LogisticRegression

# Initialize the logistic regression
log_reg = LogisticRegression(random_state = 1)

# Fit the model
log_reg.fit(X = xtrain, y = ytrain)


# **What attributes do we get from this model fit?**

# In[5]:


log_reg.classes_

log_reg.intercept_ # What does this mean?
# np.exp(log_reg.intercept_[0]) / (1 + np.exp(log_reg.intercept_[0]))

log_reg.coef_ # What does this mean?
# np.exp(log_reg.coef_[0][0])


# **What functions does the model class give us?**

# In[6]:



binary_preds = log_reg.predict(xtest)
binary_preds

soft_preds = log_reg.predict_proba(xtest)
soft_preds
# soft_preds[:, 0] # probability of 0


# Accuracy of hard classification predictions
log_reg.score(X = xtest, y = ytest) 


# **How did we do?**

# In[7]:


# Let's show the actual test data
g = sns.scatterplot(x = xtest[:, 0], y = ytest, hue = ytest == binary_preds)

# Now, let's plot our logistic regression curve
sns.lineplot(x = xtest[:, 0], y = soft_preds[:, 1])

# What is the "hard classification" boundary?
sns.lineplot(x = xtest[:, 0], y = binary_preds)
plt.axhline(0.5, linestyle = "--", color = "k") # this is what produces our classification boundary


g.set_xlabel("Age")
g.set_ylabel("CDH probability")
plt.legend(title = "Correct")

plt.show()


# Let's look at where the blue line above comes from.
# 
# Our logistic regression is formalized as follows:
# 
# For $h(x) = \beta_0 + \beta_1x$,
# 
# $p(y) = \dfrac{e^{h(x)}}{1+ e^{h(x)}}$

# In[8]:


# Let's implement the above transformation here
ypreds = np.exp(log_reg.intercept_ + log_reg.coef_*xtest) / (1 + np.exp(log_reg.intercept_ + log_reg.coef_*xtest))

# Now we can confirm that this worked
g = sns.lineplot(x = xtest[:, 0], y = ypreds[:, 0])
g.set_ylim(0, 1)
g.set_xlabel("Age")
g.set_ylabel("p(CDH)")
plt.show()

# Finally, let's look at the "linear" relationship underlying logistic regression
h = sns.lineplot(x = xtest[:, 0], y = np.log(ypreds[:, 0]/(1-ypreds[:, 0])))
h.set_xlabel("Age")
h.set_ylabel("Log odds of CDH")
plt.show()


# ## Decision Trees

# Decision trees are a form of classification that fits a model by generating successive *rules* based on the input feature values. These rules are optimized to try and classify the data as accurately as possible.
# 
# ![decision_tree](img/Decision_Tree.jpeg)
# 
# Above, the percentages are the percent of data points in each node and the proportions are the probability of survival ([Source](https://en.wikipedia.org/wiki/Decision_tree_learning)).
# 
# *Take a second to interpret this*.
# 
# Decision trees have the advantage of being super intuitive (like $k$-nearest neighbors, they're similar to how people often think about classification). 
# 
# There's a great article about how they work [here](https://towardsdatascience.com/decision-tree-classifier-explained-in-real-life-picking-a-vacation-destination-6226b2b60575) and a nice explanation of how the decision boundaries are identified [here](https://victorzhou.com/blog/gini-impurity/).

# ### Decision tree classifiers in python
# 
# You can read the decision tree classifier documentation [here](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).

# In[9]:


# Import the DecisionTreeClassifier class
from sklearn.tree import DecisionTreeClassifier


# Initialize the decision tree classifier
dtree = DecisionTreeClassifier(random_state = 1)

# Fit the model
dtree.fit(X = xtrain, y = ytrain)


# In[10]:


from sklearn import tree

tree.plot_tree(dtree)


# Whoa. 
# 
# **Decision trees can overfit data *a lot* if they aren't constrained.**
# 
# Let's try this again...

# In[11]:


dtree = DecisionTreeClassifier(
    max_depth = 1,
    random_state = 1
)

# Fit the model
dtree.fit(X = xtrain, y = ytrain)


# In[12]:


tree.plot_tree(dtree,
               feature_names = ['Age'],
               class_names = ['No CDH', 'CDH'],
               filled = True
              )


# ***What's going on here?***
# 
# - `Age <= 50.5`: This is the "rule" being used to define leaves on either side of the tree ("No" -> left, "Yes" -> right)
# - `gini = 0.453`: This refers to the "Gini impurity" of the node. Gini impurity is the loss function used to fit this tree (optimal = 0) (more on this [here](https://victorzhou.com/blog/gini-impurity/))
# - `samples = 346`: This is the number of samples in the group that the node is dividing
# - `value = [226, 120]`: This is the number of training values on the left (`values[0]`) and the right (`values[1]`) of the node
# 
# NOTE: With a depth of 1, at the very bottom, we have:
# - 170 people were correctly classified as "No CDH" with this rule (true negatives)
# - 47 people were classified as "No CDH" with this rule *incorrectly* (false negatives)
# - 56 people were classified as "CDH" with this rule *incorrectly* (false positives)
# - 73 people were classified as "CDH" with this rule *correctly* (true positives)

# Like other classifiers, the decision tree classifier lets us predict values and has functions for assessing prediction accuracy.

# In[13]:


# Accuracy on the data
dtree.score(X = xtrain, y = ytrain)
dtree.score(X = xtest, y = ytest)


# In[14]:


ypreds = dtree.predict(X = xtest)
ypreds

# Test "score" above
sum(ypreds == ytest) / len(ypreds)


# In[15]:


# The "soft classification" probabilities are just the fraction of training samples for the "true" label 
# in the leaf where this test item ended up

ypreds_soft = dtree.predict_proba(X = xtest)
ypreds_soft


# We can use the predictions as the basis for betting understanding what the tree is doing:

# In[16]:


# This reveals the cutoff(s) chosen by our decision tree! 
train_preds = dtree.predict(X = xtrain)
g = sns.scatterplot(x = xtrain[:, 0], y = ytrain, hue = ytrain == train_preds)
g.axvline(50.5)
# g.axvline(59.5)
# g.axvline(24.5)


# In[17]:


### YOUR CODE HERE

# Make a similar graph to the above with the test data


# We can also draw on the same resources that we talked about for assessing our $k$-nearest neighbors classifier

# In[18]:


from sklearn.metrics import accuracy_score, f1_score


# Test accuracy
accuracy_score(y_true = ytest, y_pred = dtree.predict(X = xtest))

# Test F1 score
f1_score(y_true = ytest,
         y_pred = dtree.predict(X = xtest),
         labels = [0, 1],
         pos_label = 1
        )


# In[19]:


from sklearn.metrics import roc_curve

# ROC curve
fpr, tpr, thresholds = roc_curve(
    y_true = ytest,
    y_score = dtree.predict_proba(X = xtest)[:, 1],
    pos_label = 1
)


sns.lineplot(x = fpr, y = tpr)
plt.axline(xy1 = (0, 0), slope = 1, c = "r")

plt.xlabel("FPR")
plt.ylabel("TPR")


# ## Support Vector Machines (SVMs)
# 
# Support Vector Machines work by trying to find a line or plane (usually in a high-dimensional space) that *maximally separates* the training labels in that space. 
# 
# The intuition for this is relatively straightforward but the implementation can get complicated!
# 
# In the plot below, the linear funtion $h_3(x_1, x_2)$ is the best way to separate our training data because it maximizes the margin on either side of the line.
# 
# SVMs try to find the equivalent of $h_3$ given some training data. This separator can be defined by the closest points in the data to the line; these are called the "support vectors". Finding the best separator usually requires mapping the training data into a high-dimensional space where it can be effectively separated.
# 
# ![svm](img/svm2.png)
# 
# ([Source](https://en.wikipedia.org/wiki/Support-vector_machine))

# ### SVMs in python
# 
# The documentation for SVMs in scikit-learn is [here](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

# In[20]:


from sklearn.svm import SVC

svm = SVC()

svm.fit(xtrain, ytrain)


# In the case of SVMs, there are class attributes that help you recover the separator that was fit.
# 
# We won't get into these but if you're interested in learning more it's good to know about!
# 

# In[21]:


# svm.intercept_
# svm.coef_ # only for 'linear' kernel
# svm.support_vectors_

# For example, we can view the items in the training set that formed the support vector
sns.scatterplot(x = xtrain[:, 0], y = ytrain)
plt.title("Training data")
plt.show()

sns.scatterplot(x = xtrain[svm.support_][:, 0], y = ytrain[svm.support_])
plt.title("Support vectors")
plt.show()


# The SVM class has a `score` function that returns the accuracy of a test set, plus prediction functions.

# In[22]:


# Percent of correct classifications
svm.score(X = xtrain, y = ytrain)
svm.score(X = xtest, y = ytest)


# In[23]:


ypreds = svm.predict(X = xtest)
ypreds


# However, soft prediction requires configuring the initial model to do soft classification (by default, SVMs are made to only do hard classification).

# In[24]:


svm_soft = SVC(probability = True) # indicate that you want the SVM to do soft classification
svm_soft.fit(X = xtrain, y = ytrain)

ypreds_soft = svm_soft.predict_proba(X = xtest)
ypreds_soft


# In[ ]:





# # Classifier Wrap-Up
# 
# This is just a sample of what's out there!
# 
# There are a number of other common classifiers you should take a look at if you're interested:
# - Naive Bayes ([here](https://scikit-learn.org/stable/modules/naive_bayes.html))
# - Discriminant analysis ([linear](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html) and [quadratic](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html))
# - Neural networks ([here](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html))
# - Random forests ([here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)) (related to decision trees)
# - Gradient boosted trees ([here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html))
# - ...
# 
# The main goal of this lecture is to show you some of the creative ways that people solve classification problems and how the scikit-learn library supports these solutions. 
# 
# This should empower you to go off and try some of these other ones on your own!

# In[ ]:




