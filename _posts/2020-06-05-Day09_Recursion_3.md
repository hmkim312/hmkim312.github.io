---
title: Day09 - Recursion 3 (Python 3)
author: HyunMin Kim
date: 2020-06-05 00:00:00 0000
categories: [Hacker Ranker, 30 Days of Code]
tags: [Factorial]
---

- URL : <https://www.hackerrank.com/challenges/30-recursion/problem>{:target="_blank"}

- Objective
    - Today, we're learning and practicing an algorithmic concept called Recursion. Check out the Tutorial tab for learning materials and an instructional video!
    - Recursive Method for Calculating Factorial
- Task
    - Write a factorial function that takes a positive integer,  as a parameter and prints the result of N (N factorial).

- Note: If you fail to use recursion or fail to name your recursive function factorial or Factorial,N you will get a score of .

- Input Format

    - A single integer,N  (the argument to pass to factorial).

- Constraints
    - 2 <= N <= 12 
    - Your submission must contain a recursive function named factorial.
- Output Format
    - Print a single integer denoting .

#### 문제풀이
- 팩토리얼 함수를 만들라는건데, python은 팩토리얼이 함수가 math에 있음


```python
import math
import os
import random
import re
import sys

# Complete the factorial function below.
def factorial(n):
    return math.factorial(n)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = factorial(n)

    fptr.write(str(result) + '\n')

    fptr.close()
```


```python
# 손으로 한번 풀어보기
def factorial(n):
    num = 1
    for i in range(1,n+1):
        num *= i
    return num
```


```python
math.factorial(3)
```




    6




```python
# 재귀함수로 해보기
def factorial_re(n):
    return n * factorial_re(n-1) if n > 1 else 1
```


```python
factorial_re(3)
```




    6




```python

```
