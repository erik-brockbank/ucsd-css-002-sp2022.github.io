#!/usr/bin/env python
# coding: utf-8

# # Lecture 4 (4/4/22)

# *Last time we covered:*
# - Data structures: lists, dictionaries, sets
# - Loops & conditions

# **Today's agenda:**
# - Datahub
# - Functions
# - numpy basics

# ### First, a little motivation

# ![jobs](img/hiring.png)

# # Datahub

# - This week's lab will be available on datahub.ucsd.edu during the first lab session today. 
# 
# - In lab, Purva will go over how to pull assignments down on datahub and help you work through the lab problems.
# 
# - This lab will be coding practice that should help a lot with this week's problem set :) and will be due next week before lab.

# # Functions

# Understanding why we use functions, how to write them, and how things can go wrong are really important to coding at all levels. I can't stress this enough! 
# 
# We won't have as much time to dedicate to functions as we should so if any of this feels shaky, please reach out!
# 
# 1. What are functions for?
# 2. How to write them
# 3. If time: scoping

# In[1]:


"""
Function cookbook
    def name_of_function([optional: parameters]):
        CODE HERE
        [optional: return X]
    
"""

# Simple example: how to add two numbers
def add_two_numbers(num1, num2):
    num1 += 1
    return num1 + num2


num1 = 5
num2 = 6
new_number = add_two_numbers(num1, num2)
new_number

num1



# In[2]:


# More tricky example: let's solve the fizzbuzz problem using a function!
# See: https://en.wikipedia.org/wiki/Fizz_buzz

# Write a function called fizz_buzz that takes in a list of numbers from 1 - 1000 and return a new list modified
# to match the rules of fizz buzz
# If a number is divisible by 3: we replace it with "fizz"
# If a number is divisible by 5: we replace it with "buzz"
# If a number is divisible by 3 *and* 5, we replace it with "fizzbuzz"
# Otherwise, we keep the same number

def fizz_buzz(number_list):
    # iterate through each number in the list (`for` loop)
    # check if the number is divisible by 3 *and* 5
    # if it is: replace with "fizzbuzz"
    # if it isn't: check if the number is divisible by 3
        # if it is: replace with "fizz"
    # check if the number is divisible by 5
        # if it is: replace with "buzz"
    # if it isn't divisible by any of these: maybe put the original number in there?
    for i in range(len(number_list)):
        if number_list[i] % 3 == 0 and number_list[i] % 5 == 0:
            number_list[i] = "fizzbuzz"
        elif number_list[i] % 3 == 0:
            number_list[i] = "fizz"
        elif number_list[i] % 5 == 0:
            number_list[i] = "buzz"

    return number_list


test_list = list(range(1, 21))
# test_list
fizz_buzz_test = fizz_buzz(test_list)
fizz_buzz_test

real_list_for_real = list(range(1, 1001))
fizz_buzz_supreme = fizz_buzz(real_list_for_real)
# fizz_buzz_supreme


# In[3]:


# If time: function scope
# Ex. passing in the fizzbuzz list, modifying in the function w/o returning it



# That's it on functions. You're going to practice writing a relatively simple function in this week's lab. This week's problem set will involve several problems that require slighly more complex functions, so if this still feels shaky by the end of this week, come on by office hours!

# # Numpy!

# ## First, what is numpy??

# Numpy is a python library that's made to support *fast* and *reliable* scientific computing. Cool. 

# In[4]:


import numpy as np


# For our purposes, there are two primary things we need to know about numpy:
# 1. It's based around using *numpy arrays* instead of traditional python lists
# 2. It offers a wide range of mathematical tools and operations over these arrays
# 
# What's the difference?
#     

# In[5]:


boring_list = [1, 2, 3, 4, 5, 6]
cool_array = np.array([1, 2, 3, 4, 5, 6])

print(type(boring_list))
print(type(cool_array))


# In[6]:


# How to square our list of numbers above?

y = [val**2 for val in boring_list] # traditional python list comprehension

y = np.square(cool_array) # numpy syntax: we do the operation over the array (no iteration, nice tidy operation)

y


# There's a lot more to say about numpy arrays and array operations than we'll have time for. 
# 
# It's worth peeking a bit at their [website](https://numpy.org/doc/stable/user/absolute_beginners.html) to get a deeper explanation of some of this stuff. 

# ## Numpy arrays: more than meets the eye

# One important and fundamental difference between numpy arrays and python lists is that arrays have to have the same kind of object inside them.

# **First**, this helps avoid some of the pitfalls of traditional python lists.

# In[7]:


confusing_list = [1, 2, '3', '4', 5.0]
confusing_list # python lets us put different kinds of stuff in a list...


# In[8]:


# ...but when we try to do operations with it, that can get us into trouble
[elem + 1 for elem in confusing_list]


# In[9]:


clear_array = np.array([1, 2, '3', '4', 5.0])
clear_array # numpy automatically chooses a format for these values


# In[10]:


clear_array = np.array([1, 2, '3', '4', 5.0], dtype='float') # or we can set one ourselves
clear_array


# In[11]:


clear_array + 1 # that lets us operate on them super easily


# **Second**, operating with numpy arrays can make our code a lot cleaner.

# In[12]:


import random

# Let's make a python list of 10000 random values between 0 and 1
x = [random.random() for _ in range(10000)]
print(x[:10])


# In[13]:


# Now, we want a new list that calculates 1/x
y = []
for value in x:
    y.append(1/value)
print(y[:10])


# In[14]:


# The code above is pretty clunky. With numpy, it's super straightforward.
array_x = np.asarray(x)

array_y = 1/array_x
array_y


# **Third**, this makes operations with numpy arrays *way faster*. Like, a lot. 

# In[15]:


import random
x = [random.random() for _ in range(10000)] 
array_x = np.asarray(x)


# In[16]:


# How long does it take to square everything in our list with normal python operations?
get_ipython().run_line_magic('timeit', 'y = [val**2 for val in x]')


# In[17]:


# What about with numpy?
get_ipython().run_line_magic('timeit', 'array_y = np.square(array_x)')


# **Why are numpy operations so different from operating on python lists?**
# 
# Numpy operations use *vectorization* on numpy arrays, meaning that the the operation is performed on the whole array at once (for the person coding), rather than having to iterate through the list ([source](https://numpy.org/doc/stable/user/whatisnumpy.html)).
# 
# ![numpy](img/numpy.png)
# 

# ## Numpy arrays -> matrices

# The numpy array generalizes to matrices and supports a lot of simple slicing and dicing. 
# 
# While we likely won't use numpy matrices as much in this course, it's worth knowing about them.

# In[18]:


a = np.ones((3, 2))
print(a)

rng = np.random.default_rng(0)
a = rng.random((3, 2))
print(a)


# In[19]:


print(a.ndim) # number of dimensions
print(a.shape) # number of values in each dimension
print(a.size) # total number of cells


# In[20]:


print(np.max(a)) # maximum operation over the whole matrix
print(np.max(a, axis = 0)) # maximum operation can specify an "axis": 0 (columns) or 1 (rows)


# ## One more thing: `pandas` is built on numpy

# It's good to be familiar with numpy syntax and functions:
# - Sometimes we'll use them in our code (esp. the super fast numpy operations)
# - And some things we do in pandas will be similar to numpy
