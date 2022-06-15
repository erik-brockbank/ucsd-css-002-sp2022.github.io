#!/usr/bin/env python
# coding: utf-8

# # Lecture 3 (4/1/22)

# *Last time we covered:*
# - Datahub and Jupyter notebooks
# - Jupyter markdown
# - Python basics: operations and data types

# **Today's agenda:**
# - Data structures: lists, dictionaries, sets
# - Loops & conditions
# - Functions
# 

# ![coaster](img/coaster.jpeg)

# # Data Structures

# ## Lists

# *What is a list?*
# - Ordered
# - Mutable
# - Variable items
# 
# Examples?

# In[1]:


# Declaring lists
foo = list() # or just: foo = []
foo = ['bar']



# Adding and removing items: append, extend, insert, remove



# Accessing items: indexing, 'in'
# Note: indexing is really flexible (ranges, -1)



# bonus: len



# Operations: sort, reverse



# Can we put lists inside of lists?? Yes. Yes we can. 



# In[2]:


# EXTRA BONUS: list comprehensions!!!
# Come back to this at the end of class if there's time...



# ## Dictionaries

# *What is a dictionary?*
# - Not ordered
# - Mutable
# - Item type matters
# 
# Examples?

# In[3]:


# Declaring dictionaries
foo = dict() # or just: foo = {}


# Adding and removing items



# Accessing items
# Note: can use `in` here as well!




# bonus: 'keys', 'values', 'items'



# Can we put a dictionary inside a dictionary?? Youbetchya! This is actually very common.



# ## Sets

# *What is a set?*
# - Not ordered
# - Mutable
# - Variable items
# 
# Examples?

# In[4]:


# Declaring sets
foo = set()
foo = set('hi how are you')
bar = set('doing fine')
foo = {1, 2, 3, 4}
bar = {2}



# Adding and removing items: add, remove



# Set operations: &, -, |



# Accessing items: bit tricky (use for loop, set operations, `pop`)
# In general, this is one reason sets aren't used very often!



# Can we put a set inside a set? ...



# # Loops & Conditions

# ## `for` Loops

# What is a `for` loop for? [aSk tHe ClaSs]

# In[5]:


# for loop syntax: `for x in some_list`, `for x in range`, `for key, val in dict.items()`



# ## `while` Loops

# What is a `while` loop for?

# In[6]:


# while loop syntax: `while ...:`



# Note: CAUTION when using while loops



# ## Conditions (`if`, `else`)

# Conditional logic plays an important role in almost any coding problem! 
# (see [this](https://en.wikipedia.org/wiki/Fizz_buzz) famous coding problem)

# In[7]:


# ifelse syntax ~inside a for loop~: fizzbuzz!



# # Functions

# Understanding why we use functions, how to write them, and how things can go wrong are really important to coding at all levels. I can't stress this enough! 
# 
# We won't have as much time to dedicate to functions as we should so if any of this feels shaky, please reach out!
# 
# 1. What are functions for?
# 2. How to write them
# 3. If time: scoping

# In[8]:


"""
Function cookbook
    def name_of_function([optional: parameters]):
        CODE HERE
        [optional: return X]
    
"""
# Example: let's turn our fizzbuzz code above into a function!



# Add parameters



# Discuss: scope
# Ex. passing in the fizzbuzz list, modifying in the function w/o returning it


