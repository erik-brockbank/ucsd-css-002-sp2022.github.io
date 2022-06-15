#!/usr/bin/env python
# coding: utf-8

# # Lecture 5 (4/6/22)

# *Last time we covered:*
# - Datahub
# - Functions
# - numpy basics

# **Today's agenda:**
# - numpy wrap-up
# - pandas basics

# # numpy wrap-up

# In[1]:


import numpy as np


# *Recall from last time...*

# **What is it?**
# 
# numpy is primarily:
# 1. A class of array objects (the `ndarray`)
# 2. A set of high-performance functions that can be executed over those arrays

# In[2]:


# 1. The ndarray
boring_list = [1, 2, 3, 4, 5, 6] # traditional python list
cool_array = np.array([1, 2, 3, 4, 5, 6]) # numpy ndarray

cool_array


# In[3]:


# 2. The functions
y = np.square(cool_array)
y
# Note this is much simpler than what we would do to perform the equivalent operation on `boring_list` above


# **Why use it?**
# 
# This represents an improvement over traditional python `list` operations for several reasons:
# - It streamlines our code
# - It's way faster

# In[4]:


# The code streamlining:

y = []
for value in boring_list: # traditional python often requires using `for` loops to execute operations on lists
    y.append(1/value)
    
y = 1/cool_array # numpy lets you apply intuitive operations to whole ndarrays at once
y


# In[5]:


# The speed:

import random
x = [random.random() for _ in range(10000)] 
array_x = np.asarray(x)

get_ipython().run_line_magic('timeit', 'y = [val**2 for val in x] # traditional python list-based approach')
get_ipython().run_line_magic('timeit', 'array_y = np.square(array_x) # numpy')


# **Combining arrays**
# 
# We can combine numpy arrays into multi-dimensional *matrices* and perform many useful operations on those matrices

# In[6]:


a = np.random.random((3, 2)) # can initialize matrices with 0s, 1s, or random numbers
print(a)


# In[7]:


# We can access individual rows or columns and individual elements using bracket [row, col] notation
a[0,]
a[:,1] # note each of these rows/columns is itself a numpy array
# type(a[0])


# In[8]:


print(np.max(a)) # maximum operation over the whole matrix
print(np.max(a, axis = 0)) # maximum operation can specify an "axis": 0 (columns) or 1 (rows)


# We'll come across numpy at various points throughout the quarter but this should be enough to get us on our feet. 
# 
# You can learn more about numpy and follow the beginner's guide on their website [here](https://numpy.org/doc/stable/user/absolute_beginners.html).
# 
# For now, it's time to switch to a new tool in our computational social science toolkit.... *pandas*

# # Pandas!

# ![panda](img/red_panda.jpeg)

# ## First, what is pandas?

# [pandas](https://pandas.pydata.org/docs/index.html) is a python library for reading, writing, and interacting with *tabular* data. 
# 
# This is convenient because a lot of data is *tabular* data. It's kind of like Excel for python (but way cooler...).
# 
# There's a *really awesome* cheat sheet [here](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf) and a series of handy tutorials [here](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html). 

# In[9]:


import pandas as pd


# In this class, we will use pandas as the basis for reading, processing, and understanding our data.
# 
# Let's get started!

# ## Reading data with pandas

# In[10]:


# mcd = pd.read_csv("../Datasets/mcd.csv")
mcd = pd.read_csv("https://raw.githubusercontent.com/UCSD-CSS-002/ucsd-css-002.github.io/master/datasets/mcd-menu.csv")


# Note: there are lots of other ways to read in data, including directly from hosted links online.

# In[11]:


get_ipython().run_line_magic('pinfo', 'pd.read_csv')


# ### What is in this dataset?

# In[12]:


mcd


# Note: printing out the whole dataset is often not the best way to look at it, and sometimes totally infeasible.
# 
# `pandas` offers several very handy tools for peeking at data.

# In[13]:


mcd.shape # number of rows and number of columns

mcd.head() # this is usually enough to get a sense of what's going on in our data
mcd.head(10) # sometimes useful to look at more rows than the default
mcd.tail() # if you're curious what kind of values or responses are at the "end" of your dataset

mcd.columns # helpful when the data has too many columns to preview (as in this data!)
mcd.index # we'll come back to this...

mcd.describe() # note: this isn't all our columns! only numeric ones
mcd.Category.value_counts() # this is the equivalent of `describe` for our categorical variables


# Note all except the last of these are operations applied directly to the pandas *data frame* `mcd` (more on this later).

# ### What can we do with this data?

# *Basic*
# - Which menu items have the most protein? Calories? Largest serving size?
# - How many items does McDonald's offer for each meal (breakfast, lunch, dinner)?
# 
# *Intermediate*
# - What are the healthiest and least healthy items on the menu?
# - What *meal* (breakfast, lunch, dinner, snack) is the most healthy or unhealthy overall? 
# 
# *Advanced*
# - Can we identify how McDonald's segments the healthy choice preferences of their customers by clustering the profiles of each menu item?

# ### Why pandas?

# Before we go any further, pause and think about how you would store this data with traditional python data structures: *a list of lists?* *Many separate dictionaries?* *A menu item class with each attribute and all items in a list?*
# 
# Think about how we would answer the questions above using traditional python operations over those data structures.
# 
# **The ways we routinely interact with data require many different kinds of (sometimes complicated) operations and data structures that can support those operations (we've already seen some of this above just to look at the data in different ways).**
# 
# We want the *flexibility of code* but the *structure of tools like Excel* to solve these problems.
# 
# This is where `pandas` comes in!

# ## How does it work?

# In[14]:


type(mcd)


# Tabular data is stored in pandas as a `DataFrame`.
# 
# A pandas data frame is essentially like a table in Excel and has similar corollaries in R, STATA, etc.
# 
# It stores data in rows organized by columns, and has some very nifty properties.

# In[15]:


# Let's look at the 'Item' column
menu_items = mcd['Item']
menu_items
type(menu_items)


# Each column in a pandas dataframe is a pandas `Series`. 
# 
# A pandas series is a lot like a numpy array, but with one additional property: the *index*.

# In[16]:


menu_items.index


# The index is a unique value used to identify each row in the series. 
# 
# You can use the index to fetch individual items in the series. 
# 
# By default, pandas just uses the row number as the index for the values in each column. 

# In[17]:


menu_items[2]


# In this way, it's a lot like a normal list or numpy array.
# 
# But, in pandas an index can use unique values of any *hashable type*.

# In[18]:


menu_cals = mcd['Calories'] # Let's fetch the `Calories` column
menu_cals
menu_cals.index # Here's the default index

# Instead, let's use each menu item as an index
menu_cals_item = pd.Series(list(mcd['Calories']), index = menu_items)
menu_cals_item


# Now, we can access items in the list using this new index!

# In[19]:


menu_cals_item['Egg McMuffin']


# What does it look like when we can look up array items with strings as keys?

# In[20]:


# We can access `index` and `values` just like dictionary keys and values
menu_cals_item.index
menu_cals_item.values


# This functions just like a dictionary in traditional python
menu_cals_lookup = dict()
for i in range(len(menu_items)):
    menu_cals_lookup[menu_items[i]] = menu_cals[i]

menu_cals_lookup
menu_cals_lookup.keys()
menu_cals_lookup.values()


# ## Take-aways

# - A pandas `DataFrame` stores tabular data in rows and columns
# - Each column is a pandas `Series` object
# - A pandas `Series` is similar to a numpy array (fixed `dtype`) but has an *index* that allows for rapid and flexible data access

# ## Accessing data in a data frame

# In the code above, we used bracket notation `dataframe['col']` to access column data.
# 
# There are a number of different ways to access columns in a data frame.
# 
# Any of these are fine, best to pick one and stick with it (and know the others exist).

# In[21]:


# Accessing individual columns
menu_items = mcd['Item']
menu_items = mcd.Item
menu_items = mcd.loc[:, 'Item']
menu_items = mcd.iloc[:,1]
menu_items


# Many of these let us access multiple columns at once:

# In[22]:


menu_subset = mcd[['Item', 'Category', 'Calories']] # Access specific columns by name
menu_subset = mcd.loc[:, 'Category':'Calories'] # Access a range of columns by name
menu_subset = mcd.iloc[:,1:4] # Access a range of columns by index
menu_subset = mcd.iloc[:,[1, 2, 5]] # Access specific columns by index
menu_subset

