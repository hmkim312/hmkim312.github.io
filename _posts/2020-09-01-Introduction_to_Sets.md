---
title: Introduction to Sets (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/py-introduction-to-sets/problem>{:target="_blank"}

- Task
    - Now, let's use our knowledge of sets and help Mickey.
    - Ms. Gabriel Williams is a botany professor at District College. One day, she asked her student Mickey to compute the average of all the plants with distinct heights in her greenhouse.

- Input Format

    - The first line contains the integer, , the total number of plants.
    - The second line contains the  space separated heights of the plants.

- Constraints
    - 0 < N <= 100

- Output Format

    - Output the average height value on a single line.

#### 문제풀이
- set을 활용하여 주어진 list를 중복제거하고, 그것을 다 더하여 평균을 구하란 이야기
- set 후 sum과 len을 활용하여 수행


```python
n = 10
arr = [161, 182, 161, 154, 176, 170, 167, 171, 170, 174]
```


```python
def average(arr):
    return sum(set(arr)) / len(set(arr))
```


```python
average(arr)
```




    169.375


