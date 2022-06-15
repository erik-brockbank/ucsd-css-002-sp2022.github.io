#!/usr/bin/env python
# coding: utf-8

# # Lecture 15 (4/29/2022)

# **Announcements**
# - Pset 4 will be released today, due next Friday 5/6

# *Last time we covered:*
# - Evaluating regression: $R^2$, out-of-sample prediction, parameter interpretation
# 
# **Today's agenda:**
# - Polynomial regression, multiple regression
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Read in and prepare the data we were using last time
mpg = sns.load_dataset('mpg')
mpg_clean = mpg.dropna().reset_index(drop = True)

mpg_clean


# # Review: Evaluating linear regression
# 
# In last lecture, we talked about three ways of checking that your regression fit the data well. 
# 1. $R^2$ *coefficient of determination*
# 2. Out of sample prediction accuracy
# 3. High confidence (and useful) parameter estimates
# 
# 
# Let's start by running through each of these in a little more detail since we didn't get much time to discuss them.

# ## $R^2$, the *coefficient of determination*
# 
# ![determination](img/Determination.png)
# 
# 
# $ R^2 = 1 - \dfrac{RSS}{TSS} $
# 
# $ RSS = \sum_{i=1}^{n}{(y_i - \hat{y_i})}^2 $
# 
# $ TSS = \sum_{i=1}^{n}{(y_i - \bar{y})}^2 $
# 
# $R^2$ ranges between 0 and 1 and can be thought of as the *percentage of variance in $y$ that our model explains*.
# 
# To understand how it works, remember that RSS is 0 when the regression *perfectly predicts our data* and RSS is equal to TSS when we just guess $\bar{y}$ for every data point $y_i$ (worst case for our regression).

# In[3]:


# The scikit-learn LinearRegression class surfaces a function called `score` that computes R^2
from sklearn.linear_model import LinearRegression

# Format values
x_vals = np.array(mpg_clean['weight']).reshape(len(mpg_clean['weight']), 1)
y_vals = np.array(mpg_clean['horsepower'])

# Fit regression
mod = LinearRegression().fit(X = x_vals, y = y_vals)

rsq_mod = mod.score(X = x_vals, y = y_vals) # R^2 value
rsq_mod


# Last time, we showed how to calculate the $R^2$ value by hand using the LinearRegression `predict` function. 
# 
# If you're feeling hazy on $R^2$, I recommend going back to the code from that lecture and going through the part where we calculate $R^2$.
# 

# ***

# ## Out of sample prediction
# 
# ![train](img/train.jpeg)
# 
# 
# **Motivation** 
# 
# If our model is the right fit to our data, it should predict other data from the same underlying distribution or generative process pretty well.
# 
# **How to check this**
# 
# There are a lot of ways to test out of sample data which we'll get into in more detail on Monday, but the high-level approach is almost always:
# 1. *Randomly* select a subset of your original data (20-25%) and set it aside as *test data*. The remaining data is your *training data*.
# 2. Fit your model to the *training data only*. 
# 3. See how well your fitted model predicts the *test data*. Compare it to the predictions on the training data with something like *Mean Squared Error* (MSE). 
# 4. Often, repeat steps 1-3 in various ways (more on that later).
# 
# **Comparing train and test performance**
# 
# Step 3 above is critical. One common approach is to use *Mean Squared Error* (MSE):
# 
# $ MSE = \dfrac{1}{n - 2} \sum_{i=1}^{n}{(y_i - \hat{y_i})}^2 = \dfrac{1}{n - 2} \sum_{i=1}^{n}{\epsilon_i}^2 $
# 
# This tells you, on average, how close your model was to the true value across all the data points (the $n-2$ is specific to linear regression where we have two parameters, $\beta_0$ and $\beta_1$, so $n-2$ is our degrees of freedom).

# In[4]:


from sklearn.model_selection import train_test_split # Use the sklearn `train_test_split` to make this easy
from sklearn.metrics import mean_squared_error # Use the sklearn `mean_squared_error` for quick MSE calculation

# Randomly sample 25% of our data points to be test data
xtrain, xtest, ytrain, ytest = train_test_split(x_vals, 
                                                y_vals, 
                                                test_size = 0.25,
#                                                 random_state = 500
                                               )

# Fit the model on the training data
mod_tr = LinearRegression().fit(X = xtrain, y = ytrain)

# Generate model predictions for the test data
mod_preds_test = mod_tr.predict(X = xtest)

# Compare MSE for the model predictions on train and test data
mod_preds_train = mod_tr.predict(X = xtrain)

mse_train = mean_squared_error(y_true = ytrain, y_pred = mod_preds_train)
mse_train # Note this divides by n rather than n-2 but that's not super important for our purposes

mse_test = mean_squared_error(y_true = ytest, y_pred = mod_preds_test)
mse_test

print("Training MSE: {} \nTest MSE: {}".format(mse_train, mse_test))


# Just for fun, try running the code above several times and look at how different the values are. 
# 
# *More on this next week...*

# ***

# ## Parameter estimates
# 
# ![right](img/right.png)
# 
# The criteria above are mostly concerned with whether we're doing *a good job predicting our $y$ values* with this model.
# 
# In many cases, part of what we're concerned with isn't just how well we predict our data, but what kind of relationship our model estimates between $x$ and $y$. 
# - How large or small is the slope?
# - How confident are we in the estimate?
# 
# To assess this, we typically compute confidence bounds on the parameter estimates (95% confidence interval or standard error) and compare them to a null value of 0 using $t$ tests.
# 
# **Linear regression parameter estimates are most useful when they are high confidence and significantly different from 0.**
# 
# The sklearn `LinearRegression` class doesn't include functions for this sort of analysis, but other tools like the `statsmodels` regression class do.

# In[5]:


import statsmodels.formula.api as smf

# Fit the model
results = smf.ols('horsepower ~ weight', data = mpg_clean).fit()

# View the results
results.summary()


# ***

# # Problems with simple linear regression
# 
# ![corr_ex](img/corr_ex.png)
# Disclaimer: this figure (from wikipedia) shows *correlation* values associated with these datasets, but the limitations of correlation in capturing these patterns holds for linear regression as well.

# ## Polynomial regression: non-linear relationship between $x$ and $y$
# 
# Non-linear data can take all kinds of forms, though there are probably a few that are most common.
# 
# Let's take a look at a simple example from our cars dataset:

# In[6]:


sns.lmplot(data = mpg_clean, x = "horsepower", y = "mpg")


# Does this data have a linear relationship between $x$ and $y$? Seems like it might be more complicated.
# 
# Enter: polynomial regression!

# ### Polynomial regression: overview
# 
# Polynomial regression is just like linear regression except that instead of fitting a linear function to the data, we fit higher degree polynomials. 
# 
# Previously, our simple linear regression model assumed that our data $(x_i, y_i)$ could be described as:
# 
# $y_i = \beta_0 + \beta_1 x_i + \epsilon_i$
# 
# The OLS process estimates values for $\beta_0$ and $\beta_1$ that correspond to a straight line that minimizes $\epsilon_i$.
# 
# With polynomial regression, we extend this basic model to include functions of the form:
# 
# $y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \epsilon_i$ for *degree 2* polynomial regression,
# 
# $y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \beta_3 x_i^3 + \epsilon_i$ for *degree 3* polynomial regression, 
# 
# $y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \beta_3 x_i^3 + \ ... \ + \beta_n x_i^n + \epsilon_i$ for *degree n* polynomial regression.
# 
# Even though this seems much more complex, polynomial regression uses the same *Ordinary Least Squares* (OLS) parameter estimation as simple regression. You can think of simple linear regression as a special case of polynomial regression. 
# 
# **This gives us immense flexibility to fit more complex functions to our data.** Some of the data illustrated at the top of this section can *only* be modeled using more complex polynomials (see example below as well).
# 
# **CAUTION**: most of the time you *don't* need polynomials to capture your data. Bad things happen when you use them for data that doesn't have an underlying non-linear structure. More on this on Monday.
# 
# ![poly](img/Poly.png)

# ### Polynomial regression in python
# 
# We can use the numpy `polyfit` library to fit 2nd and 3rd order polynomials to this data
# (Note: this is probably the simplest method, but there's code to use the familiar scikit learn approach as well below).

# In[7]:


# We can fit higher order polynomial functions to our data rather than just a linear function
deg1_fits = np.polyfit(mpg_clean.horsepower, mpg_clean.mpg, 1)
deg2_fits = np.polyfit(mpg_clean.horsepower, mpg_clean.mpg, 2)
deg3_fits = np.polyfit(mpg_clean.horsepower, mpg_clean.mpg, 3)

p1 = np.poly1d(deg1_fits)
p2 = np.poly1d(deg2_fits)
p3 = np.poly1d(deg3_fits)


# In[8]:


# What do the functions fitted above predict for our data?

preds = mpg_clean.loc[:, ('horsepower', 'mpg')] # 

preds['deg1_pred'] = p1(preds['horsepower'])
preds['deg2_pred'] = p2(preds['horsepower'])
preds['deg3_pred'] = p3(preds['horsepower'])

preds

preds_long = preds.melt(
    id_vars = ['horsepower', 'mpg']
)
preds


# In[9]:


# First, our original data
sns.scatterplot(data = preds_long,
                x = 'horsepower',
                y = 'mpg',
                color = 'm',
                alpha = 0.1
               )

# Now add in our lines
sns.lineplot(data = preds_long,
             x = 'horsepower',
             y = 'value',
             hue = 'variable'
            )


# Here's the solution using scikit learn; it's a bit more complicated, though it does let you keep using the `LinearRegression` class 

# In[10]:


from sklearn.preprocessing import PolynomialFeatures

x_vals = np.array(mpg_clean['horsepower']).reshape(len(mpg_clean['horsepower']), 1)
y_vals = np.array(mpg_clean['mpg'])
preds = mpg_clean.loc[:, ('horsepower', 'mpg')]

# Simple linear model
mod1 = LinearRegression().fit(x_vals, y_vals)

# 2nd order polynomial
poly2 = PolynomialFeatures(degree = 2, include_bias = False) # need `include_bias` = False
x2_features = poly2.fit_transform(x_vals)
mod2 = LinearRegression().fit(x2_features, y_vals)


# 3rd order polynomial
poly3 = PolynomialFeatures(degree = 3, include_bias = False)
x3_features = poly3.fit_transform(x_vals)
mod3 = LinearRegression().fit(x3_features, y_vals)


mod2.intercept_
# mod2.coef_
# mod3.coef_


# In[11]:


# Add predictions for each model so we can view how it does
preds['deg1_pred'] = mod1.predict(x_vals)
preds['deg2_pred'] = mod2.predict(x2_features)
preds['deg3_pred'] = mod3.predict(x3_features)

preds


# In[12]:


preds_long = preds.melt(
    id_vars = ['horsepower', 'mpg']
)
preds

# First, our original data
sns.scatterplot(data = preds_long,
                x = 'horsepower',
                y = 'mpg',
                color = 'm',
                alpha = 0.1
               )

# Now add in our lines
sns.lineplot(data = preds_long,
             x = 'horsepower',
             y = 'value',
             hue = 'variable'
            )


# In[13]:


# Let's check the R^2 values for these models to see what kind of improvement we get
# (more on this next week)


# ***

# ## Multiple regression: multiple predictors for $y$
# 
# Another basic scenario that arises when predicting a continuous variable (probably more commonly than polynomial regression) is having *multiple predictors*.
# 
# Let's take a look at an intuitive example:

# In[14]:


gap = pd.read_csv("https://raw.githubusercontent.com/UCSD-CSS-002/ucsd-css-002.github.io/master/datasets/gapminder.csv")

# Let's keep just some of the variables (note for pset!)
gap_subset = gap.loc[gap['year'] == 2007, ('country', 'year', 'lifeExp', 'pop', 'gdpPercap')]

# Add log population
gap_subset['logPop'] = np.log10(gap_subset['pop'])
gap_subset['logGdpPercap'] = np.log10(gap_subset['gdpPercap'])
gap_subset


# In the last problem set, you generated a graph that predicted life expectancy as a function of income, with information about population and region available as well. 
# 
# ![gap](img/gapminder.png)
# 
# The graph suggests that life expectancy is strongly predicted by income, while population may not play such an important role. 
# 
# Let's test that here!
# 
# What that amounts to asking is:
# 
# **Can we predict life expectancy using both income *and* population better than we could only using one of those variables?**

# In[15]:


sns.scatterplot(data = gap_subset, 
                x = "logGdpPercap", # x1 
                y = "lifeExp",
                color = "r"
               )
plt.show()

sns.scatterplot(data = gap_subset, 
                x = "logPop", # x2 
                y = "lifeExp",
                color = "b"
               )
plt.show()


# ### Multiple regression: overview
# 
# Multiple regression is like linear regression except that we assume our dependent variable $y_i$ is *jointly* predicted by multiple independent variables $x_1$, $x_2$, ..., $x_n$, as in the example above.
# 
# As noted above, our simple linear regression model assumes that our data $(x_i, y_i)$ has the following form:
# 
# $y_i = \beta_0 + \beta_1 x_i + \epsilon_i$
# 
# With multiple regression, we now extend this model to include multiple predictors:
# 
# $y_i = \beta_0 + \beta_1 x_{i,1} + \beta_2 x_{i,2} + \ ... \ + \beta_n x_{i,n} + \epsilon_i $
# 
# In most cases, multiple regression once again uses the same *Ordinary Least Squares* (OLS) parameter estimation as simple regression. However, interpreting the parameter estimates is a little less straightforward.
# 
# *How would we interpret $\beta_0 = 1$, $\beta_1 = 2$, $\beta_2 = 3$*?

# ### Multiple regression in python
# 
# To run our multiple regression, we can use the scikit `LinearRegression` class with just a few modifications to our simple regression code.
# 
# I've also included the statsmodels code below as well so we can look at the statistics more closely!

# In[16]:


# scikit learn approach
x_vals = np.array(gap_subset[['logGdpPercap', 'logPop']]) # Note: this syntax is important!
x_vals = x_vals.reshape(len(gap_subset), 2)
x_vals

y_vals = np.array(gap_subset['lifeExp'])
y_vals

mod = LinearRegression().fit(X = x_vals, y = y_vals)

mod.intercept_
mod.coef_


# In[17]:


# How well does our regression do?
mod.score(X = x_vals, y = y_vals)


# Using the statsmodels regression class, we can view statistical tests on our parameter fits

# In[18]:


multiple_reg = smf.ols('lifeExp ~ logGdpPercap + logPop', data = gap_subset).fit()

# View the results
multiple_reg.summary()

