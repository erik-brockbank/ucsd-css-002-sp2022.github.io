# Debugging

***This content largely written by Ed Vul.***

Your first attempt at writing a computer program will contain errors (bugs). It's not about you, this is true for everyone, including your instructors and professional programmers. Learning to program requires learning how to fix errors in your code. This process is called "debugging". Here we provide some accumulated wisdom about debugging. Fortunately, the steps to fixing the bug yourself, searching the internet for help, or asking someone else for help are the same.

## Identify what kind of error it is.

There are many kinds of errors your code might contain. Debugging begins by figuring out what kind of bug you are dealing with.

- **Code does not execute**: These bugs will yield some kind message from the Python interpreter describing the error with a key word of the form ***SomeError*** (e.g., ValueError, SyntaxError, IndentationError, TypeError, ZeroDivisionError, etc.). These errors may be further divided into:

    - **Syntax errors**, which arise when the interpreter does not know how to parse the code you have written (e.g., failure to close parentheses, or quotes, incorrect indentation, missing colon, missing comma, etc.). Python indicates that an error has occurred in its attempt to parse the code, and points out where it *thinks* the error occurred. However, since Python could not parse your code, it might be incorrect in identifying the location of the error.

    - **Runtime exceptions**, which arise when the interpreter can parse the code you have written, but cannot do something you have told it to do (e.g., add a string and an integer, divide by 0, refer to a non-existant index/key/variable/method)

- **Code executes, but does the wrong thing**: These bugs will not yield any kind of error message -- the program will execute without the computer thinking that anything is amiss, but the program will do something other than what you intended. These kinds of bugs are much harder to diagnose and fix! When trying to fix these sorts of bugs try to figure out: did you tell the computer what to do incorrectly? or did you tell it to do the wrong thing?


## Distill to a minimal, reproducible example

Fixing bugs, or asking others for help debugging your code, requires that you *isolate* the bug. Doing so amounts to identifying a minimal reproducible set of conditions that reproduces the problematic behavior.

1. Find the line that causes the error.

1. Find the *type* and *values* of the variables that are being called on that one line. (print-debugging is helpful to do this -- insert `print()` statements into your code to print out the various variables, and their types, in the middle of execution)

1. Write down some completely self-sufficient code that sets the relevant variables to those values, and runs the line of code that produces the error.

If you have successfully isolated the bug, you should be able to paste those minimal lines of code to a brand new notebook, restart the kernel, and produce the bug again. If you cannot do that, that means that you have failed to isolate the bug, go back to step 1.

The process of isolating a bug will often yield the solution (e.g., because you will realize that variables have different values than you expected), but even if it does not, it is a necessary step to asking someone for help. For more on this process see the [stackoverflow how to](https://stackoverflow.com/help/minimal-reproducible-example).

### Example

```Python
a = input('enter number a: ')
b = input('enter number b: ')
print('a plus b is...')
print(a+b)
```

If you run this, and enter the numbers 76 and 12 the answer you get out is 7612, which is... not expected. We can insert some print statements to figure out what's going on with the variables. What are their types and values?

```Python
a = input('enter number a: ')
b = input('enter number b: ')
print(type(a), a)
print(type(b), b)
print(type(a+b), a+b)
```

Which (upon the entry of 76 and 12) yields the output:
```
<class 'str'> 76
<class 'str'> 12
<class 'str'> 7612
```

So we now know the *types* and *values* of `a` and `b` that produce the surprising behavior, and we can omit the `input` statements. Furthermore, we know the exact value and type of the surprising answer.

```Python
a = '76'
b = '12'
a + b
```

We could even reduce this example further and omit the variables altogether:

```Python
'76' + '12'
```

## Explain to a rubber ducky

Explain the minimal, reproducible example to a [rubber ducky](https://en.wikipedia.org/wiki/Rubber_duck_debugging) or some other patient, non-judgmental entity. Explain line by line what you expect the line to do, and what each line is actually doing, and how your expectations deviate from reality. Sometimes, explaining to a rubber ducky might be necessary just to *isolate* the error. Explain to rubber duckies as often as you need.

Once you have explained the minimal reproducible example out loud to a rubber ducky, write down a concise version of that explanation. That is your explanation of the bug and your confusion.

An example, for the following problematic code:

```Python
a = '76'
b = '12'
a+b
```
**Explanation:** `a` is set to the string '76'; `b` is set to the string '12'; I expect `a+b` to sum the numbers 76 and 12 and yield 88, but instead it yields the string 7612.

### Things to explain to the duck

When explaining what your code is doing, and what you intended it to do, you should try to ask yourself the following questions (and explain them to the duck):

- Are you telling the computer what to do incorrectly, or are you telling the computer to do the wrong thing? e.g., are you using the wrong commands, or is the procedure you described incorrect?
 Your very first few days of programming will involve very simple procedures, while you do not understand the commands, so you will likely tell the computer to do the wrong thing. As you advance and start writing more complicated programs, you will quickly transition to using the correct commands, but telling the computer to do the wrong thing.

- What assumptions are you making, and what do you not understand? In the example above, we are making certain assumptions about the behavior of the `+` operator. If we learn that `'straw' + 'berry'` yields `'strawberry'`, we might have some idea about what kind of false assumptions we are making.


## Google for the answer

Successfully googling requires some basic strategies, a bit of understanding of the code, and some familiarity with the relevant websites on the internet.

- Add "in python" to the end of a question you want to ask. (e.g., "how do I convert a string to a number in python")

- Identify the *standard* parts of the command/operation (rather than your particular names/values). Use those standard parts in your search.
 For instance, in the example above "a + b gives 7612 in python" is a bad search term that does not yield anything particularly useful. The problem is that the variable names `a` and `b`, as well as their values `'76'` and `'12'`, were arbitrarily chosen by me, and I should not expect anyone on the internet to know anything about them. However, the things that are standard and general are that they are *strings* that contain *numbers* which I tried to *add*. So a fruitful google search would be something like "add number strings in python". If I knew a bit more terminology, my search term might be even better: "adding two number strings concatenates rather than sums in python". (these will get you to an answer.)

- did you get an error? What was the exact error (e.g., SomethingError). What was the standard function / operation that triggered it? What were the *types* of variables you were using? e.g., "TypeError add int and str in Python"

- Know which sites to look in. add "site:stackoverflow.com" to your search query to search only on stackoverflow or "site:python.org" to search only the official documentation.

- Know how to use the common sites. e.g., on stackoverflow look for the accepted answer, consider the highly upvoted comments, look for the function documentation on python.org, etc.

- Use "Find in page" (usually command-f or control-f) to look for keywords on long pages.

## Checklist to debug and to ask debugging questions

Here is a checklist for asking a debugging question. Provide answers to each of these questions:

**1. What are you trying to do?**
> e.g., I am trying to add two numbers in strings

**2. What kind of error is it?**
> e.g., The code executes but yields an unexpected answer.

**3. What is the minimal reproducible example?**
e.g., `'76'+'12'`

**4. What is the rubber-duck explanation of the minimal example: what did you expecct the code to do, and did it do instead?**
> e.g., I expect that adding the string '76' and the string '12' would yield 88, but instead it yields the string '7612'.

**5. Describe what you have tried to do to solve the problem.** (i.e., I googled X, Y, and Z, and looked at these sites)
> e.g., I googled for
> "add number strings in python site:stackoverflow.com", and
> "adding two number strings concatenates rather than sums in python", and
> "add two numbers as strings python", and looked at the top results
> [here](https://stackoverflow.com/questions/11999228/python-adding-number-to-string),
> [here](https://www.tutorialspoint.com/program-to-add-two-numbers-represented-as-strings-in-python),
> [here](https://docs.python.org/3/tutorial/introduction.html),
> [here](https://www.tutorialspoint.com/program-to-add-two-numbers-represented-as-strings-in-python).
