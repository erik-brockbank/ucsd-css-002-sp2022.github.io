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
foo = ['bar', 'hello']

css_students = ['ancen', 'jiayi', 'adriana', 'michelle', 'tanvi']
css_students


# Adding and removing items: append, extend, insert, remove
css_students.append('glenn')
css_backrow = ['andres', 'starlee', 'advaith']
css_students.extend(css_backrow)
css_students.insert(0, 'new_student')

# css_students.remove('new_student')

# Accessing items: indexing, 'in'
if ('new_student' in css_students):
    css_students.remove('new_student')

css_students  

css_students[0]
css_students[1:]
css_students[1:-2]


# Note: indexing is really flexible (ranges, -1)



# bonus: len
len(css_students)


# Operations: sort, reverse
css_students.append('5')
css_students.sort()

# css_students.reverse()
css_students

# Can we put lists inside of lists?? Yes. Yes we can. 
css_students.append(['more_students_yay'])

css_students


# In[2]:


# EXTRA BONUS: list comprehensions!!!
# Come back to this at the end of class if there's time...



# ## Dictionaries

# *What is a dictionary?*
# - key-value store
# - Not ordered
# - Mutable
# - Item type matters
# 
# Examples?

# In[3]:


# Declaring dictionaries
foo = dict() # or just: foo = {}
foo = {'blah': 6, 'beep': 0, 'sigh': 3}

# Adding and removing items
foo['woo'] = 4

new_key = 'womp'
new_val = 10
foo[new_key] = new_val


del foo['womp']

foo


# Accessing items
# Note: can use `in` here as well!

if 'woo' in foo:
    print("found woo!")


# bonus: 'keys', 'values', 'items'

foo.keys()
foo.values()
foo.items()

# This is a common and helpful way to loop through dictionaries!
for key, val in foo.items():
    print(key)
    print(val)


# Can we put a dictionary inside a dictionary?? Youbetchya! This is actually very common.
colors = {'R': 'red', 'O': 'orange'}
foo['color_lookup'] = colors

del foo['color_lookup']
# foo

foo['boop'] = 0
foo

foo['boop'] = 2
foo


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
# bar = set('doing fine')
foo = {1, 2, 3, 4}
bar = {2}



# Adding and removing items: add, remove

foo.add('b')
foo.remove(2)

foo

# Set operations: &, -, |

dir(foo)


# Accessing items: bit tricky (use for loop, set operations, `pop`)
# In general, this is one reason sets aren't used very often!



# Can we put a set inside a set? ...



# # Loops & Conditions

# ## `for` Loops

# What is a `for` loop for? [aSk tHe ClaSs]

# In[5]:


# for loop syntax: `for x in some_list`, `for x in range`, `for key, val in dict.items()`

# iterate through lists
for student in css_students:
    print(student)

# print(range(0, 10))
# iterate through numbers
for num in range(0, len(css_students)):
    print(css_students[num])

# looping through dictionary key/values    
for key, val in foo.items():
    print(key)


# ## `while` Loops

# What is a `while` loop for?

# In[6]:


# while loop syntax: `while ...:`

index = 0
while index < 5:
    print(index)
    index += 1


# Note: CAUTION when using while loops



# ## Conditions (`if`, `else`)

# Conditional logic plays an important role in almost any coding problem! 
# (see [this](https://en.wikipedia.org/wiki/Fizz_buzz) famous coding problem)

# In[7]:


# ifelse syntax ~inside a for loop~: fizzbuzz!

nums = []
for num in range(101):
    new_num = num
    if num % 3 == 0 and num % 5 == 0:
        new_num = 'fizzbuzz'
    elif num % 3 == 0:
        new_num = 'fizz'

    nums.append(new_num)

nums


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


