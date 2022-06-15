#!/usr/bin/env python
# coding: utf-8

# # Lecture 10 (4/18/22)

# **Announcements**
# - Problem set 2 *due tonight at 11:59pm*
# - Change of grading basis deadline 4/22 (shooting to have pset 1 & 2, labs 1 & 2 graded by then)

# *Last time we covered:*
# - graphing with matplotlib
# - graphing best practices
# - graphing with seaborn
# 
# **Today's agenda:**
# - Cleaning and structuring data with pandas

# # Cleaning data: principles and tools

# In[1]:


import pandas as pd
import numpy as np


# ## What are the sorts of things we (typically) don't want in our data?

# ...
# 
# 

# ## What do the things we don't want in our data look like in python?

# ### Duplicates

# In[2]:


# Duplicates: what do they look like?

df = pd.DataFrame({
    "sample_int": np.random.randint(low = 1, high = 10, size = 10)
})

df
# df[df.sample_int == 1]


# Duplicates are hard to detect without a more careful search because they may look like totally normal data otherwise.

# ### Missing data

# In[3]:


# Missing data: what does it look like?

df.iloc[0] = np.nan
df.iloc[1] = None

df # is this what we expected?


# In[4]:


df['nans'] = np.nan
df['nones'] = None

df


# Note: sometimes missing data can take other forms, for example empty strings.
# 
# This is especially true when your data has had some intermediate processing like being stored as a csv.

# ### Outliers

# In[5]:


# Outliers: what do they look like?

df.iloc[9, 0] = 400

df


# Be careful here!
# 
# Not every unexpected data point is an outlier...

# ## Aside: why do we want to detect these things in our data?

# In[6]:


# A simple illustration

df

np.mean(df.sample_int)

# np.sum(df.sample_int) / len(df.sample_int)

# len(df.sample_int)
# np.sum(df.sample_int) / 8



np.median(df.sample_int) # uh oh...


# Take-away: these things can impact our analyses in lots of ways, which is why we want to find them!

# ## How do we check for stuff we don't want in our data?

# ### Duplicates

# In[7]:


# Duplicates

df[df.duplicated()] # just show the info that's duplicated after the first occurrence

# df[df.duplicated(keep = False)] # show all duplicated rows
# df


# ### Missing values

# In[8]:


# Missing values

# Pandas tools: `pd.isna` and `pd.isnull`
pd.isna(df['sample_int'])
df[pd.isna(df['sample_int'])]

df[pd.isnull(df['sample_int'])] # Note: this is the same as `pd.isna` above


# In[9]:


# Numpy tools: `np.isnan` to detect NaNs
df[np.isnan(df['sample_int'])]

# But watch out! 
np.isnan(np.nan)
# np.isnan(None) # that's weird...


# ### Outliers

# This is one area where graphing our data can be really helpful!

# ![anscombe](img/anscombe.png)

# ## How do we get rid of stuff we don't want in our data?
# 
# Our options are:
# 1. Remove obervations (or otherwise ignore them)
# 2. Fill in the observations (missing values only)

# ### Duplicates

# With duplicate data, the best thing is most often to remove it (unless the duplication is expected)

# In[10]:


df_no_dupes = df.drop_duplicates(ignore_index = True) # Note: we need this `ignore_index = True`

df_no_dupes


# ### Missing values
# 
# With missing values, it may be best to drop missing values, or fill them in with a reasonable value.
# 
# 
# More info on missing value handling [here](https://pandas.pydata.org/docs/user_guide/missing_data.html).

# **Dropping missing values**: why might we do that?

# In[11]:


# Dropping NAs

df_no_na = df.dropna() # can specify an `axis`: 0 for columns, 1 for rows containing missing values

df_no_na # What happened here??


# In[12]:


df_no_na = df.dropna(how = "all") 
# how = "all" will drop rows where *all* values are NaN
# how = "any" (default) will drop rows where *any* column value is NaN

df_no_na # that's better


# **Filling in missing values**: Why might we do that instead? (and what to put there?)

# The pandas `fillna` function provides a few convenient ways to fill in missing data

# In[13]:


# Can provide a specific number

df_no_na = df.fillna(0) 

# df_no_na = df.fillna(df.mean()) # note this can be handy
# df_no_na = df.fillna(np.nanmedian(df['sample_int'])) 

df_no_na


# In[14]:


# Can also use the previous or next value in the sequence (handy for time series data)

df.iloc[4, 0] = np.nan

df

df_no_na = df.fillna(method = 'pad',
                     limit = 1 # optional limit argument says how many consecutive times to do this
                    )

df_no_na


# Finally, pandas also provides tools for using *interpolation* to guess the missing value. 
# 
# We won't get into that here, but you can check out [this page](https://pandas.pydata.org/docs/user_guide/missing_data.html#interpolation) if you're curious.

# ### Outliers

# When you have data that includes large outliers, it may be okay to remove them from your data, but you should do this with caution! 
# 
# As a general rule, only remove outliers when you have a good justification for doing so.
# 
# *What constitutes a good justification?*
# - If for whatever reason your analysis is only interested in values within a particular range
# - Or, more often, if you have strong reason to suspect that a datapoint was *generated differently* than the rest (for example, somebody falling asleep at the keyboard)
# 
# In these cases, you typically set a criterion and remove rows that meet that criterion.
# 

# In[15]:




np.std(df.sample_int) # can specify e.g., values > X standard deviations above the mean
thresh = np.mean(df.sample_int) + 3*np.std(df.sample_int)

np.nanquantile(df.sample_int, [0.95]) # or, similarly, the top N% of data points
thresh = np.nanquantile(df.sample_int, [0.95])[0]

thresh

df_no_outliers = df[df.sample_int <= thresh]

df_no_outliers


# 

# In[ ]:




