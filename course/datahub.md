# Datahub assignments

This is a guide to submitting assignments on datahub.

## Overview

**Technical overview:** Assignments are [jupyter notebooks](https://jupyter-notebook.readthedocs.io/en/stable/) served from UCSD's [datahub](https://datahub.ucsd.edu), and automatically graded by [nbgrader](https://nbgrader.readthedocs.io/en/stable/).

**Practical overview:** To complete an assigment you will need to:
1. **Log in to [datahub](https://datahub.ucsd.edu):** If you are enrolled in the class, you should be able to log in using your UCSD credentials (username@ucsd.edu, and your password)
1. **Launch this class environment:** select "CSS 2 - Data/Model Programming..." and click "Launch Environment".
1. **Fetch assignment:** click on "Assignments" and click "Fetch" to download to your account space.
1. **Open assignment:** Click on assignment name in downloaded list, to expand assignment files, then click on the main file in the assignment to open the assignment notebook.

1. **Complete assignment** follow instructions in the notebook, remove `raise NotImplementedError()`, save regularly.
1. **Check your work:** re-read questions and instructions; restart kernel and rerun all cells; look over results; save.
1. **Submit:** Back in the assignment list, click "validate" to check that you are passing all visible tests. Once validated, click "Submit"! You can submit multiple times (see cautionary note below).

## Cautionary points:

- **DO NOT**: change the name of this file, try to change or delete test/`assert` cells, try to change other read only cells, copy/paste or delete cells (but you can make new cells as you like).

- **You MUST submit the notebook to receive assignment credit. We cannot grade assignments that have not been submitted.**

- You can submit at any time, but **we grade your most recent submission**. This means that **if you submit an updated notebook after the submission deadline, it will be marked as late**.

- We may have *hidden* tests for answers that will only run during official grading -- so make sure your code does what it is supposed to: passing visible assert tests is not sufficient for getting the right answer.

- Remember to **remove the `raise NotImplementedError()` line** in code cells you answer!

## Tips & Tricks

The following are a couple tips & tricks that may help you if you get stuck on anything.

- Run cells and test cell asserts to check your answers (if a test fails, read the error for clues about how to fix it).

- Consult our [debugging guide](debugging.md)

- **Print variables:** you can (and should) print and check variables as you go. This allows you to check what values they hold, what types they are, and fix things if anything unexpected happens.

- **Restart the Kernel:** If you run cells out of order, you can end up overwriting things in your namespace. If things seem to go weird, a good first step is to restart the kernel, which you can do from the kernel menu above. (also 'Restart & Run All' before submitting, to make everything runs properly in order).
