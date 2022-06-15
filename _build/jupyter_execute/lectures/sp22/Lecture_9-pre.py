#!/usr/bin/env python
# coding: utf-8

# # Lecture 9 (guest) - Data Visualization with Seaborn
# 
# ***Author: Umberto Mignozzetti***

# ## Seaborn
# 
# **`Seaborn`** is a data visualization library built on the top of `matplotlib`. It was created by [Micheal Waskon at the Center for Neural Science, New York University](https://joss.theoj.org/papers/10.21105/joss.03021).
# 
# **`Seaborn`** has all the attributes of the `matplotlib` library (it is a child class), making it considerably easy to plot data using Python.
# 
# We will learn some of these plots in this class and a few customizations. More about `Seaborn` can be found in [here](https://seaborn.pydata.org).
# 
# Below you can find a list of functions that we can use to plot data on `Seaborn`.
# 
# ![alt image](https://seaborn.pydata.org/_images/function_overview_8_0.png)

# In[1]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # This is how you import seaborn

# Datasets

## Political and Economic Risk Dataset
# Info on investment risks in 62 countries in 1992
# courts  : 0 = not independent; 1 = independent
# barb2   : Informal Markets Benefits
# prsexp2 : 0 = very high expropriation risk; 5 = very low
# prscorr2: 0 = very high bribing risk; 5 = very low
# gdpw2   : Log of GDP per capita
perisk = pd.read_csv('https://raw.githubusercontent.com/umbertomig/seabornClass/main/data/perisk.csv')
perisk = perisk.set_index('country')

## Tips Dataset
# Info about tips in a given pub
# totbill : Total Bill
# tip     : Tip
# sex     : F = female; M = male
# smoker  : Yes or No
# day     : Weekday
# time    : Time of the day
# size    : Number of people
tips = pd.read_csv('https://raw.githubusercontent.com/umbertomig/seabornClass/main/data/tips.csv')
tips = tips.set_index('obs')


# And here is what we have in these datasets:

# In[2]:


perisk.head()


# In[3]:


tips.head()


# ## Plotting Data 101
# 
# The best way to explore the data is to plot it. However, not all plots are suitable for the variables we want to describe. Starting with a single variable, the first question is what type of variable we are talking about?
# 
# Types of variables:
# 
# - `Quantitative` variables: represent measurement.
#     
#     + `Discrete`: number of children, age in years, etc.
#     
#     + `Continuous`: income, height, GDP per capita, etc.
# 
# - `Categorical` variables: represent discrete variation
# 
#     + `Binary`: voted for Trump, smokes or not, etc.
#     
#     + `Nominal`: species names, a candidate supported in the primaries, etc.
#     
#     + `Ordinal`: schooling, grade, risk, etc.
# 
# For each variable type, there are specific descriptive stats and plots. Below, see an example of the difference between using the `right` and `wrong` descriptive stats for continuous and binary variables.

# In[4]:


# Summary stats for a continuous variable (good)
perisk['gdpw2'].describe()


# In[5]:


# Frequency table for a continuous variable (bad)
perisk['gdpw2'].value_counts()


# In[6]:


# Summary stats for a binary variable (bad)
perisk['courts'].describe()


# In[7]:


# Frequency table for a binary variable (good)
perisk['courts'].value_counts()


# ## Univariate Plots
# 
# *Univariate plots* are plots for single variables.
# 
# ### Quantitative Variables: Histograms
# 
# Starting with numerical variables, one suitable plot is the *histogram*. It breaks the numerical values into brackets and counts how many values are within each bracket.
# 
# The syntax is:
# 
# ```
# sns.displot(data = the_data_frame,
#     x = 'the_variable',
#     kind = 'hist',
#     kde = [..True or False..], 
#     rug = [..True or False..],
#     bins = [..number of bins..], 
#     stat : [..{"count", "density", "probability"}..],
#     [..among others..])
# ```
# 
# Let's plot a histogram for the Log of GDP per capita (`gdpw2`)?

# In[8]:


# My code here


# ### Customizations
# 
# We can easily customize the entire plot:
# 
# 1. **Main title**: `plt.title('title here')`
# 
# 2. **X-axis title**: `g.set_xlabels('text')` or `plt.xlabel('text')`
# 
# 3. **Y-axis title**: `g.set_ylabels('text')` or `plt.ylabel('text')`
# 
# 4. **Style**: 'white', 'dark', 'whitegrid', 'darkgrid', and 'ticks'. Usage: `sns.set_style('stylename')`
# 
# 5. Remove the spine: `g.despine(left = True)`
# 
# 6. **Current Palette + display the palette**: `sns.palplot(sns.color_palette())`
# 
# 7. **Which palettes**: `sns.palettes.SEABORN_PALETTES` and to change, use `set_palette('palette')`
# 
# 8. **Save figure**: instead of `plt.show()` use `plt.savefig('figname.png', transparent = False)`.
# 
# 9. **Context**: set the context between 'paper', 'notebook', 'talk', and 'poster'. Use `sns.set_context('context here')`
# 
# There are even more customization that we can do. Please check the [seaborn documentation](https://seaborn.pydata.org/tutorial/function_overview.html) for more details.

# In[9]:


# My code here


# **Exercise**: Using the histogram, describe the variables `totbill` and `tip` in the `tips` dataset.

# In[10]:


## Your answers here


# ### Categorical Variables: Countplot
# 
# Countplots are suitable for displaying categorical variables. 
# 
# The syntax is:
# 
# ```
# sns.catplot(
#     data = the_data_frame,
#     x = 'the_variable', 
#     kind = 'count')
# ```
# 
# Let's check the risk of expropriation in each of the countries in 1992.

# In[11]:


# My code here


# All the customizations that we learn apply here as well. We can use them to prettify this plot. 
# 
# However, since the scale is out of order, we can change the order of the x-axis values using the `order` parameter. 
# 
# Even more, for `ordinal` data, it is customary to use a sequential color scheme, i.e., it gets darker as we increase the categories. 
# 
# We can use several palettes:
# 
# 1. `Blues`
# 2. `Greys`
# 3. `PuRd`: Light Purple to Dark Red
# 4. `GnBu`: Light Green to Dark Blue
# 
# Among others. The syntax to create the color scheme is:
# 
# ```
# sns.set_palette(
#     sns.color_palette("color_scheme", # If want revert add '_r'
#                       [..number_of_colors or as_cmap=True..])
# )
# ```
# 
# For more about color palettes, please check [here](https://seaborn.pydata.org/tutorial/color_palettes.html).

# In[12]:


# My code here


# **Exercise**: Do a countplot for the days (`day`) in the `tips` dataset.

# In[13]:


## Your answers here


# ## Bivariate Plots
# 
# Univariate plots are excellent. But in reality, most of the exciting questions in science come from combinations of multiple variables (e.g., cause and effect, correlations, relationships, etc). 
# 
# For two variables' plots there are three combinations:
# 
# - **discrete x discrete**: mosaic plot
# 
# - **discrete x continuous**: several useful types
# 
# - **continuous x continuous**: scatterplots
# 
# ### Discrete x Discrete Variables: Mosaicplot
# 
# The mosaic plot gives an idea of how the ratio of one variable changes when we change another variable. For instance, one empirical question that we can ask about the `perisk` dataset is:
# 
# **Do countries with independent courts have less corruption than countries without independent courts?**
# 
# The code to test this idea takes two steps. First, we need to prep the data. Then, we plot the data using the `kind = 'bar'` in the `catplot` function.
# 
# We need to create a table with cumulative values for the two variables we want to study to prep the data. Here is an example of how to do that:
# 
# ```
# tab = pd.crosstab(df.v1, df.v2, normalize='index') # 1: Crosstab
# tab = tab.cumsum(axis = 1).\     # 2: Cummulative sum
#       stack().\                  # 3: Stack the results
#       reset_index(name = 'dist') # 4: Reset the indexes
# tab
# ```
# 
# Then, we need to plot the results using `catplot`:
# 
# ```
# sns.catplot(data = tab,
#             x = 'v1', # More variation here
#             y = 'dist', # Proportions
#             hue = 'v2', # Less variation here
#             # Comment hue_order if not displaying well
#             hue_order = tab.v2.unique()[::-1], 
#             dodge = False,
#             kind = 'bar')
# plt.show()
# ```
# 
# *Full disclosure*: A function exists that builds mosaic plots in one line of code. However, I find the results very ugly. You can Google `mosaic plot in python` and check that yourself.

# In[14]:


# My code here: prepping the data


# In[15]:


# My code here: doing the plot


# **Exercise**: Do the number of smokers (variable `smoker`) vary by the weekday (`day`)?

# In[16]:


## Your answers here


# ### Discrete x Continuous Variables: Boxplots, Swarmplots, Violinplots
# 
# Suppose we want to test whether the data distribution varies based on a categorical variable. For example:
# 
# **Do you think that having an independent judiciary affects the GDP per capita of a country?**
# 
# We can check if this hypothesis makes sense by looking into the distribution of GDP per capita and segmenting it by the type of judicial institution.
# 
# The syntax for building these plots is almost the same as making a single boxplot. The difference is that you add the categorical variable to one of the axes:
# 
# ```
# sns.catplot(
#     data = data_set, 
#     x = 'categorical_variable',
#     y = 'continuous_variable',
#     kind = 'box') # Or 'violin', 'swarm', 'boxen', 'bar'..
# ```

# In[17]:


# My code here


# **Exercise**: Are the tips from smokers higher than tips from non-smokers? (the idea is that smokers would compensate non-smokers for the externality caused) Check that in the `tips` dataset.

# In[18]:


## Your answers here


# ### Continuous x Continuous Variables: Scatterplots and Regplots
# 
# To plot two continuous variables, one against the other, we can use two functions. First, we can use the `relplot` function if we want to explore the relationship without fitting any trend line. The syntax is the following:
# 
# ```
# sns.relplot(data = data_set,
#             x = 'independent_axis_continuous_variable',
#             y = 'dependent_axis_continuous_variable',
#             hue = 'optional_categorical_to_color',
#             kind = 'scatter')
# ```
# 
# And an excellent version of it, with distribution plots on the top and the left, can be built using the `jointplot` function:
# 
# ```
# sns.jointplot(data = data_set,
#               x = 'independent_axis_continuous_variable',
#               y = 'dependent_axis_continuous_variable',
#               hue = 'optional_categorical_to_color',
#               kind = 'scatter') # Or 'scatter', 'kde', 
#                                   'hist', 'hex', 'reg', 
#                                   'resid'
# ```
# 
# If you want to add a trend line, it is better to use `lmplot` (instead of 'reg' in the plot above). The syntax is the following:
# 
# ```
# sns.lmplot(data = data_set,
#     x = "total_bill", 
#     y = "tip", 
#     hue = "smoker",
#     logistic = ..False or True.., # Logistic fit for discrete y
#     order = ..polynomial order.., # Polynomial degree
#     lowess = ..False or True..,   # Lowess fit
#     ci = ..None..)                # Remove conf. int.
# ```

# In[19]:


# My code here


# **Exercise**: Are the tips related with total bill in the `tips` dataset?

# In[20]:


## Your answers here


# **Great job!!!**

# ## After-class extras
# 
# Excellent job learning `seaborn`! It is an easy-to-use yet powerful package to generate lovely plots.
# 
# Next, you should take a look at the following packages to keep developing your skills:
# 
# - [`plotnine`](https://plotnine.readthedocs.io/en/stable/index.html#): Implements the ggplot *grammar of graphs* in python
# 
# - [`cartopy`](https://github.com/SciTools/cartopy): Package to make maps in python.
# 
# - [`plotly`](https://plotly.com): Builds interactive graphs in python (and other languages). Check also the [`dash`](https://dash.plotly.com/introduction) for plotly in python.
# 
# Now, try the extra exercises below to sharpen your learning.

# In[21]:


## Extra Datasets

## Political Information Dataset
# ANES 2000 Political Information based on interviews
# polInf          : Political Information
# collegeDegree   : College Degree
# female          : Female
# age             : Age in years
# homeOwn         : Own house
# others...
polinf = pd.read_csv('https://raw.githubusercontent.com/umbertomig/seabornClass/main/data/pinf.csv')
pinf_order = ['Very Low', 'Fairly Low', 'Average', 'Fairly High', 'Very High']
polinf['polInf'] = pd.Categorical(polinf.polInf, 
                                  ordered=True, 
                                  categories=pinf_order)

## US Crime data in the 1970's
# Data on violent crime in the US
# Muder: number of murders in the state
# Assault: number of assaults in the state
# others...
usarrests = pd.read_csv('https://raw.githubusercontent.com/umbertomig/seabornClass/main/data/usarrests.csv')


# In[22]:


polinf.head()


# In[23]:


usarrests.head()


# ### Exercises
# 
# 1. (Univariate) In the `polinf` dataset, make a count plot of the variable `polInf`. Imagine you want to use this for a talk, so adjust the context. Change the x-axis label and title to appropriate descriptions of the data. (Hint: to rotate the axis tick labels, use `plt.xticks(rotation=number_degree_of_your_choice)`)
# 
# 2. (Univariate) In the `polinf` dataset, make a histogram of the variable `age`. (Hint: set the context back to `notebook` before starting)
# 
# 3. (Bivariate) Do you think political information varies with a college degree? Check that using the `polinf` dataset!
# 
# 4. (Bivariate) Do you think political information varies with age? Check that using the `polinf` dataset!
# 
# 5. (Bivariate) Do you think there is a correlation between `Murder` and `Assault`? Check that using the `usarrests` dataset!
# 
# 6. (Challenge: Multivariate) There are four continuous indicators in the `usarrests` dataset: `Murder`, `Assault`, `UrbanPop`, and `Rape`. Do you think you can build a scatterplot matrix? The documentation is in [here](https://seaborn.pydata.org/examples/scatterplot_matrix.html).

# In[24]:


## Your answers here

