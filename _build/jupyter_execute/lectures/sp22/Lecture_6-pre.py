#!/usr/bin/env python
# coding: utf-8

# # Lecture 6 (4/8/22)

# *Last time we covered:*
# - numpy continuation
# - pandas basics

# **Today's agenda:**
# - Processing data with pandas

# In[1]:


import pandas as pd
import numpy as np


# # Creating new dataframes

# It's worth noting: there are *many* different ways to do operations like dataframe creation in pandas. 
# 
# This is probably the most intuitive but you may come across others!

# In[2]:


# First let's initialize some data
# Typically, we want to think of our data as being lists of the stuff in each column
uc_schools = np.array(["Berkeley", "San Diego", "Los Angeles", "Santa Barbara", "San Francisco", 
                       "Irvine", "Davis", "Riverside", "Santa Cruz", "Merced"])
uc_founded = np.array([1868, 1960, 1919, 1909, 1864, 1965, 1905, 1954, 1965, 2005])
uc_students = np.array([45057, 42875, 45742, 26314, 3132, 35220, 40031, 25548, 19161, 8847])


# Now let's put it in a dataframe!
uc_data = pd.DataFrame({ # start by declaring a new data frame
    "School": uc_schools, # each column name is a dictionary key and the list of column data points is the value
    "Date founded": uc_founded,
    "Number of students": uc_students
})

uc_data


# We will likely find ourselves reading in data more often than creating new dataframes, but it's occassionally useful and good to know how to do it!

# # Adding data to existing dataframes

# ## Adding columns (common!)

# We may often find ourselves adding columns to a dataframe, e.g., creating columns that represent our existing data in a new way

# In[3]:


# One option: when we have new data to add
uc_undergrads = np.array([31814, 33343, 31543, 23349, 0, 30222, 31162, 22055, 17207, 8151])

uc_data['Undergraduates'] = uc_undergrads # Use bracket syntax to declare a new column

uc_data


# In[4]:


# A second option: when we want to process existing data and form a new column

uc_data = uc_data.assign(Undergraduate_pct = uc_data['Undergraduates'] / uc_data['Number of students'])

uc_data # Note what happens if we don't do the re-assignment above

# For this sort of processing, we can also use similar syntax to the above (try it yourself!)
# but `assign` gives us some additional flexibility


# ## Adding rows

# In[5]:


# Let's say the UC system decides it's long overdue to build a campus in Lake Tahoe near the Nevada border. 
# We want to add some projected data

# First, we make our new row into a dataframe of its own, with matching columns
uc_tahoe = pd.DataFrame({
    "School": ["Lake Tahoe"],
    "Date founded": [2022] # Note we don't need to know all the column info here
})

uc_tahoe


# In[6]:


# Next, we use `concat` to add it to the previous dataframe
uc_data = pd.concat([uc_data, uc_tahoe])

uc_data


# # Processing data: filtering, grouping, summarizing

# First, note that some operations can be done on our dataframe without having to use filtering or grouping

# In[7]:


uc_data = uc_data.sort_values('Date founded', ascending = True)
uc_data

# Note this doesn't change index at far left!


# In[8]:


print(uc_data.max())
print(uc_data['Number of students'].min())


# ## Filtering: fetch rows that meet certain conditions

# ### Filtering operations given by pandas

# We can do some kinds of filtering with built-in operations like `nlargest`

# In[9]:


top_3_students = uc_data.nlargest(3, 'Number of students')
top_3_students


# ### Logical filtering

# Most often, we filter by setting logical criteria over certain column values

# In[10]:


# Get quantiles of undergraduate sizes
q = uc_data['Undergraduates'].quantile([0.25, 0.75])
q[0.75]

# Fetch uc_data rows where uc_data['Undergraduates'] is > the 75th percentile value above
large_undergrads = uc_data[uc_data['Undergraduates'] > q[0.75]] 
large_undergrads


# *How does this work?*

# In[11]:


uc_data['Undergraduates'] > q[0.75]


# So in the code above, we're basically saying "fetch the rows where this condition evaluates to `True`".
# 
# Note this can get much more complicated...

# In[12]:


# What's going on here?
max_undergrad = uc_data[uc_data['Undergraduates'] == uc_data['Undergraduates'].max()]
max_undergrad


# ### Why is this useful?

# Let's ask: what is the average number of undergraduates in the schools with the most students overall

# In[13]:


q = uc_data['Number of students'].quantile([0.80])

# What's going on here?
uc_data[uc_data['Number of students'] >= q[0.8]]['Undergraduates'].mean()


# Here, we applied some statistics to a filtered subset of our data. 
# 
# Imagine we wanted the same average for *each quartile* of student numbers, instead of just the top 20%.
# 
# This kind of thing is pretty common: this is where grouping comes in. 

# ## Grouping and summarizing: analyze and graph your data

# First let's read in some more complex data.
# 
# What's going on with this data?

# In[14]:


# pokemon = pd.read_csv("../Datasets/Pokemon.csv")
# Use this code in class
pokemon = pd.read_csv("https://raw.githubusercontent.com/UCSD-CSS-002/ucsd-css-002.github.io/master/datasets/Pokemon.csv")
pokemon


# Let's say we want to know which `Type 1` group has the most `HP` on average
# [any guesses?]
# 
# One solution: 
# 1. Look at each individual `Type 1` value in our data
# 2. Then, one by one, filter the data to match each unique `Type 1` value
# 3. Then, compute the average `HP` in that filtered data
# 4. Then save it somewhere else to keep track of 
# 5. Then see which one is largest
# 
# This seems pretty tough...

# Let's start with something simpler: how many of each kind of `Type 1` pokemon are there?

# In[15]:


pokemon.groupby(['Type 1']).size().reset_index()


# Above, we just asked pandas to tell us the value of `size()` applied to each unique group of `Type 1` pokemon.
# 
# Can we do the same thing but for more complex operations than `size()`? You bet!

# In[16]:


pokemon.groupby(['Type 1']).agg( # .agg is our friend here!
    mean_hp = ('HP', np.mean) # this defines a new statistic over each grouping. Apply `np.mean` to the `HP` column
).reset_index()


# We don't need to stop there. We can group by multiple variables and add multiple metrics!

# In[17]:


type1_hp_summary = pokemon.groupby(
    ['Type 1', 'Legendary'] # Note we're now grouping by each combination of Type 1 and Legendary
).agg(
    mean_hp = ('HP', np.mean), # And we're compiling multiple statistics here
    min_hp = ('HP', np.min),
    max_hp = ('HP', np.max)
).reset_index()

type1_hp_summary


# Now we can apply the filtering we discussed above for example.

# In[18]:


type1_hp_summary.nlargest(5, 'mean_hp')


# ## Let's practice!

# In[ ]:





# In each `Generation`, how many different `Type 1` and `Type 2` types are there?
# 
# [HINT: use the `nunique` summary function]

# In[19]:


# Write here














# SOLUTION
# pokemon.groupby('Generation').agg(
#     type1_types = ('Type 1', 'nunique'),
#     type2_types = ('Type 2', 'nunique')
# ).reset_index()


# Make a new column called `Composite_force` that's the average of each pokemon's `Attack`, `Defense`, and `Speed` values.
# 
# Next, for each `Type 1` type, what is the maximum of this new column?

# In[20]:


# Write here
















# SOLUTION
# pokemon = pokemon.assign(
#     Composite_force = (pokemon['Attack'] + pokemon['Defense'] + pokemon['Speed'])/3
# )

# max_force = pokemon.groupby(['Generation']).agg(
#     max_force = ('Composite_force', np.max)
# ).reset_index()

