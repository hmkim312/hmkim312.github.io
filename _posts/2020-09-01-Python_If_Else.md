---
title: Python If-Else (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/py-if-else/problem>{:target="_blank"}

- Task
    - Given an integer, , perform the following conditional actions:

        - If  is odd, print Weird
        - If  is even and in the inclusive range of  to , print Not Weird
        - If  is even and in the inclusive range of  to , print Weird
        - If  is even and greater than , print Not Weird
- Input Format
    - A single line containing a positive integer, .

- Constraints
    - 1 <= n <= 100

- Output Format
    - Print Weird if the number is weird; otherwise, print Not Weird.


```python
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input().strip())

    if n %2 != 0:
        print('Weird')
    elif n <= 5:
        print('Not Weird')
    elif n <= 20:
        print('Weird')
    else:
        print('Not Weird')
```

     100


    Not Weird