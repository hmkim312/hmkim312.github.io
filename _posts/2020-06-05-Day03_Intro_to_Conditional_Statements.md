---
title: Day03 - Intro to Conditional Statements (Python 3)
author: HyunMin Kim
date: 2020-06-05 00:00:00 0000
categories: [Hacker Ranker, 30 Days of Code]
tags: [If Else]
---

- URL : <https://www.hackerrank.com/challenges/30-conditional-statements/problem>{:target="_blank"}

- Objective
    - In this challenge, we're getting started with conditional statements. Check out the Tutorial tab for learning materials and an instructional video!

- Task
    - Given an integer, n , perform the following conditional actions:

        - If  is odd, print Weird
        - If  is even and in the inclusive range of 2 to 5 , print Not Weird
        - If  is even and in the inclusive range of 6 to 20, print Weird
        - If  is even and greater than 20, print Not Weird
    - Complete the stub code provided in your editor to print whether or not n is weird.

- Input Format

    - A single line containing a positive integer, n.

- Constraints
    - 1 <= n <=100
- Output Format
    - Print Weird if the number is weird; otherwise, print Not Weird.


```python
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    N = int(input())
```

     24



```python
N = 24
if N % 2 != 0:
    print('Weird')
elif N <= 5:
    print('Not Weird')
elif N <= 20:
    print('Weird')
else:
    print('Not Weird')
```

    Not Weird



```python
N = 3
if N % 2 != 0:
    print('Weird')
elif N <= 5:
    print('Not Weird')
elif N <= 20:
    print('Weird')
else:
    print('Not Weird')
```

    Weird
