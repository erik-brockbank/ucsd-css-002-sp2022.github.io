#!/usr/bin/env python
# coding: utf-8

# # Lecture 2 (3/30/22)
# 
# *Getting back in the swing of things...*
# - Datahub and jupyter notebooks
# - Python review: operations + data types
#     
# <img src="img/patrick.jpeg" align="left" width=400>

# In[ ]:





# In[ ]:





# # First up: jupyter notebooks

# This is a "markdown" cell!
# 
# You can format markdown cells in lots of ways:

# In[ ]:





# # BIIIIGGG HEADERS
# 
# ## Medium Headers
# 
# ### smol headers
# 
# #### extra smol headers??
# 
# ##### extra smol italic headers???
# 

# You can also format regular text:
# 
# *FORTUNE* favors the **BOLD**.

# Markdown is powerful, but it will never be:
# - bullet
# - proof
# 
# Indeed, some would say markdown's:
# 1. days
# 2. are
# 3. numbered

# A few other tricks:
#     
# You can add hyperlinks to the [World Wide Web](https://en.wikipedia.org/wiki/Alan_Turing)

# You can also embed images:
#     
# ![ExampleImage](img/florence.jpeg)

# In fact, markdown essentially compiles to HTML, so if you have experience coding in HTML, you can do all kinds of nifty things.
# 
# 
# <span>Hello, world.</span>
# <div align="center">Watch out!</div>
# <div align="right">Over here!</div>

# ## What's the point of markdown cells anyway?
# 
# Good question! Jupyter notebooks are made to simplify the process of sharing and collaborating with code. 
# 
# Often, it's helpful to annotate what your code is doing, describe results in more detail, or just organize your code into different sections. 
# 
# Markdown cells give you the tools to do that.
# 

# *...But what about the code?* 🤖 🤖 🤖 🤖 🤖

# In[1]:


# You can also run python code in jupyter notebooks!
# That's actually their main job!


# In[2]:


print("hello, world")


# *So what kind of code will we run?*

# # Time for... python review!

# **The goal of this section is to refresh your memory for the basics of python coding.**
# 
# This should be a review for most of you! 
# 
# However, the more practice you get with the python basics, the easier everything else this quarter will be.
# 
# If you have questions about any of this, it's important to clear them up now :)

# ## Operations

# In[3]:


# Arithmetic

4 + 5 # addition

4 - 5 # subtraction

4 * 5 # multiplication

4 / 5 # division

4**2 # exp


# In[4]:


# Booleans ('and', 'or', 'not')

True 
False

True and False

True or False

True and not False


# In[5]:


# Comparison (>, <, ==, !=)

4 < 5

4 > 5

'5' == 5

4 != 5


# In[6]:


# Assignment

foo = 5

foo = '5'


# bonus: assignment shorthand (+=)

foo = 5

foo *= 2 # +=, -=, *= /=

foo


foo = 'a'
foo += '5'
foo


# In[7]:


# Special operation: 'in'

my_string = 'hi hello it is Wednesday'

'Wednesday' in my_string


# ## Data types

# ### Integers and floats

# In[8]:


# Integers and floats

my_int = -3
my_float = 3.0

type(my_int)
type(my_float)



# Operations: math!

my_int * 3

my_int *= 3

my_int


# ### Strings

# In[9]:


# Strings
foo = 'hi'
bar = "hello"


foo = 'hi it\'s Wednesday' # escape single quotes inside quotes
foo = "Bob said, \"My name is Bob\""
foo


# dir(foo)


# Operations
# find()
# replace()
# split()

# foo.find('My')
# bar = foo.replace('Bob', 'Steve')
# foo.split(' ')




# bonus: string indexing

foo[0]
foo[5]


# ### Identifying data types and operations
# *A rose by any other name...*

# In[10]:


foo = "bar"

print(type(foo)) # What kind of thing is this?
print(dir(foo)) # What operations can we do with this?





# ## Data structures

# ### Lists

# *What is a list?*

# In[11]:


# Declaring lists

foo = ['bar']



# Adding and removing items
append()
extend()
insert()
remove()



# Accessing items: indexing, 'in'



# bonus: len



# Operations: sort, reverse



# ### Dictionaries

# *What is a dictionary?*

# In[12]:


# Declaring dictionaries
foo = {}
foo = dict()


# Adding and removing items



# Accessing items



# bonus: 'keys', 'values', 'items'



# ### Sets

# *What is a set?*

# In[13]:


# Declaring sets
foo = set()
foo = set('hi how are you')
bar = set('doing fine')
foo = {1, 2, 3, 4}
bar = {2}

# Adding and removing items


# Set operations: &, -, |



# Accessing items: bit tricky...

