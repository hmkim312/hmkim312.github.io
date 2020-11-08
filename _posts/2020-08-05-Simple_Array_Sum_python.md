---
title: Simple Array Sum (Python 3)
author: HyunMin Kim
date: 2020-08-05 00:00:00 0000
categories: [Hacker Ranker, Problem Solving]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/simple-array-sum/problem>{:target="_blank"}

Given an array of integers, find the sum of its elements.

For example, if the array ar = [1,2,3] 1 + 2+ 3 = 6 so return 6.

#### Function Description

- Complete the simpleArraySum function in the editor below. It must return the sum of the array elements as an integer.

- simpleArraySum has the following parameter(s):

- ar: an array of integers

#### Input Format

- The first line contains an integer, , denoting the size of the array.
- The second line contains  space-separated integers representing the array's elements.


```python
ar = [1,2,3]
```


```python
def simpleArraySum(ar):
    num = 0
    for i in ar:
        num+= i
    return num
```
