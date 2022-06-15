#!/usr/bin/env python
# coding: utf-8

# # Lecture 11 (4/20/2022)

# **Announcements**
# - Problem set 3 coming out today, will be due *next Wednesday* 4/27
# 

# *Last time we covered:*
# - Data cleaning with python (duplicates, missing data, outliers)
# 
# **Today's agenda:**
# - *Wide* versus *long* data
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Wide and Long Data

# > "Like families, tidy datasets are all alike but every messy dataset is messy in its own way."
# ~ Hadley Wickham

# ## What is *wide* data?
# 
# ...

# When we interact with data that's made to be read by people, it's most often in *wide* format. 
# 
# The definition of wide data can be a little hard to pin down but one rule of thumb is that *wide* data spreads multiple observations or variables across columns in a given row.
# 
# | y | x1 | x2 | x3 |
# | -- | -- | -- | -- |
# | 1 | a | b | c |
# | 2 | d | e | f |
# | ... | ... | ... | ... |

# Here's some data I made up about average temperatures in five US cities over three consecutive years:

# In[2]:


cities = pd.DataFrame({
    "City": ["San Diego", "Denver", "New York City", "Los Angeles", "San Francisco"],
    "2010": [75, 60, 55, 65, 70],
    "2011": [77, 63, 58, 67, 72],
    "2012": [77, 62, 56, 67, 71]
})

cities


# This data can also be presented with *year* as our variable of interest and each city as a column:

# In[3]:


years = pd.DataFrame({
    "Year": [2010, 2011, 2012],
    "San Diego": [75, 77, 77],
    "Denver": [60, 63, 62],
    "New York City": [55, 58, 56],
    "Los Angeles": [65, 67, 67],
    "San Francisco": [70, 72, 71]
})

years


# Both of these are pretty easy to read and pretty intuitive. 
# 
# **What kind of questions can we answer most easily with each dataframe?**
# 
# `cities`: 
#     
# `years`: 
# 
# 

# Note: this is easiest to illustrate with *time sequence* data, but most data can be toggled around this way to some degree:

# In[4]:


students = pd.DataFrame({
    "Student": ["Erik", "Amanda", "Maia"],
    "Math": [90, 95, 80],
    "Writing": [90, 85, 95]
})

students


# In[5]:


classes = pd.DataFrame({
    "Subject": ["Math", "Writing"],
    "Erik": [80, 95],
    "Amanda": [95, 85],
    "Maia": [80, 95]
})

classes


# The first table makes it easier to ask questions like "which student performed best?", while the second is easier for asking questions like "are these students better at math or writing?"

# **Self-quiz:** do the above examples give you an okay intuition for what *wide* data looks like?

# ## So what's the problem?

# 1. First, the exercise above suggests that for different kinds of questions, we need to format our data in different ways. That seems onerous...
# 
# 2. Second, even though tables like this make these data easy to read as humans, answering questions about the data when it's formatted like this can be difficult and inefficient. 

# *Using the data below, how do we figure out which city was hottest on average (using our python skills)?*

# In[6]:


cities


# In[7]:


# CODE HERE


# Notice that we have to do this by calculating an average row by row. Seems inefficient.
# 
# Can we do any better with our `years` dataframe?

# In[8]:


years


# In[9]:


# CODE HERE


# *Using the data below, how do we decide which year had the highest recorded temperature across these cities?*

# In[10]:


years


# In[11]:


# CODE HERE


# Yikes 😬

# **Self-quiz:** is it clear how data that's easy to read in wide format can be kind of tricky to interact with when trying to analyze it in python?

# ## What do we do about this? Answer: tidy (long) data!

# With *long* or *tidy* data, every observation gets its own row, with columns indicating the variable values that correpond to that observation.
# 
# The *wide* table at the beginning of the previous section looked like this:
# 
# | y | x1 | x2 | x3 |
# | -- | -- | -- | -- |
# | 1 | a | b | c |
# | 2 | d | e | f |
# | ... | ... | ... | ... |
# 
# Compare the table above to this one:
# 
# | y | variable | value |
# | -- | -- | -- |
# | 1 | x1 | a |
# | 1 | x2 | b |
# | 1 | x3 | c | 
# | 2 | x1 | d | 
# | 2 | x2 | e | 
# | 2 | x3 | f | 
# | ... | ... | ... |

# Here's a concrete example with the student data above. 
# 
# In wide form, it looked like this:

# In[12]:


students


# In *tidy* form, it looks like this:
# 

# In[13]:


tidy_students = pd.DataFrame({
    "Student": ["Erik", "Erik", "Amanda", "Amanda", "Maia", "Maia"],
    "Subject": ["Math", "Writing", "Math", "Writing", "Math", "Writing"],
    "Score": [90, 90, 95, 85, 80, 95]
})

tidy_students


# **Self-quiz:** is it clear how the *tidy* data here differs from *wide* data?
# 
# If you want to go into the weeds on this, [here's](https://www.jstatsoft.org/article/view/v059i10) a paper by the inventor of `tidyverse`, a large library in R with many similar functions to `pandas`. 

# ## So what does *tidy* data do for us?

# The tidy data in the previous examples are harder to read and harder to interpret in the ways we often want to think about tabular data. 
# 
# *So how does this help us?*

# **Summary**
# - Tidy data avoids the pitfalls of having to reformat our data for different kinds of questions (usually)
# - Tidy data enforces structure so there isn't confusion about how best to represent our data (there may be multiple wide formats but usually only one tidy format) -> *best practice*
# - Tidy data is easier to interact with and analyze with code
# - Tidy data lets us take advantage of the *vectorization* that numpy, pandas, and other modern coding languages employ to make calculations super speedy

# **Example**
# 
# Let's go through a simple example with the temperature data above. 
# 
# Here's the original wide dataframe:

# In[14]:


cities


# Here it is in tidy format:

# In[15]:


tidy_cities = pd.DataFrame({
    "City": ["San Diego", "San Diego", "San Diego", 
             "Denver", "Denver", "Denver", 
             "New York City", "New York City", "New York City", 
             "Los Angeles", "Los Angeles", "Los Angeles", 
             "San Francisco", "San Francisco", "San Francisco"
            ],
    "Year": [2010, 2011, 2012, 
             2010, 2011, 2012, 
             2010, 2011, 2012, 
             2010, 2011, 2012, 
             2010, 2011, 2012
            ],
    "Temp": [75, 77, 77,
             60, 63, 62,
             55, 58, 56,
             65, 67, 67,
             70, 72, 71
            ]
})

tidy_cities


# Now, let's return to our original question: *which city was the hottest on average during this time?*

# In[16]:


# CODE HERE


# That was pretty easy. 
# 
# And under the hood, pandas `groupby` means that we compute the average temperature using vectorization rather than calculating row by row as we did in the solution above. 
# 

# What about our second question: *which year had the highest recorded temperature?*

# In[17]:


# CODE HERE


# Okay, that was also pretty easy.
# 
# So, this is far from an exhaustive survey of wide versus tidy/long data, but should give you a flavor for why this distinction is useful.

# **Self-quiz:** do the examples above make it pretty clear why tidy data makes our lives simpler, clearer, and easier for coding / analysis?

# ## Pandas helps you convert data easily

# Lots of data in the real world comes in wide form or requires some re-shuffling to get into tidy format.
# 
# If you're working with a dataset that isn't in tidy form, it's almost always a good first step. 
# 
# We'll quickly review the tools that `pandas` has for toggling data formats.
# 
# Much more info about this [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html#).
# 

# ### Converting from wide to long with `melt`

# First, let's turn to a familiar dataset: the `gapminder` data.
# 
# *Is this data in tidy form?*

# In[18]:


gap = pd.read_csv("https://raw.githubusercontent.com/UCSD-CSS-002/ucsd-css-002.github.io/master/datasets/gapminder.csv")

gap
# gap.shape # note the size. Things are about to change...


# Let's move the "observations" (`lifeExp`, `pop`, and `gdpPercap`) to their own rows using [`melt`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html#pandas.melt):

# In[19]:


gap_tidy = gap.melt(
    id_vars = ["Unnamed: 0", "country", "continent", "year"], # columns to keep in each row
    value_vars = ["lifeExp", "pop", "gdpPercap"], # columns to be moved into their own rows
    var_name = "measure", # name of the column that will store the "value_vars" column names
    value_name = "value" # name of the column that will store the "value_vars" column values
)

gap_tidy # take a look at the data. Is this what you expected?

# gap_tidy.shape # note how many rows we added with this


# What can we do with this?
# 
# Quick example! 
# 
# (think about how we would do the below with our data in wide format)

# In[20]:


gap_tidy.groupby(
    ['country', 'measure']
)['value'].mean().reset_index()


# ### Converting from long to wide with `pivot`

# But wait! I thought we wanted our data in tidy format???
# 
# The [`pivot`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pivot.html) function makes it easy for us to convert to wide format when it's convenient. 

# In[21]:


gap_wide = gap_tidy.pivot(
    index = "year", # column to be treated as the index
    columns = ["measure", "country"], # columns to be spread out into their own columns for each value
    values = "value" # value to be inserted in each new column
)

gap_wide
# gap_wide.shape # this is super condensed


# In[22]:


# We access data in the above by making our way down the hierarchical columns
gap_wide.columns

gap_wide['lifeExp']['Australia'][2002]


# In[23]:


# This can make things like plotting this data a little easier (no need to filter ahead of time)
g = sns.scatterplot(x = gap_wide['gdpPercap']['United States'],
                    y = gap_wide['lifeExp']['United States']
                   )

g.set_xlabel("Average income ($ GDP / capita)")
g.set_ylabel("Avg. life expectancy (years)")
g.set_title("Income and life expectancy in the US")


# ## Bonus: `stack` and `unstack`
# 
# A really clear overview [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html#reshaping-by-stacking-and-unstacking)
# 
# 

# In[24]:


gap_stack = gap_wide.stack("country")

gap_stack
# gap_stack.columns


# In[25]:


# gap_stack['pop']
# gap_stack[gap_stack['year'] == 2007]


# In[26]:


gap_unstack = gap_stack.unstack("year")
gap_unstack

