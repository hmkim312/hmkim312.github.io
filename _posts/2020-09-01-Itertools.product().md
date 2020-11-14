---
title: Itertools.product() (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/itertools-product/problem>{:target="_blank"}
- This tool computes the cartesian product of input iterables.
- It is equivalent to nested for-loops.
- For example, product(A, B) returns the same as ((x,y) for x in A for y in B).

- Sample Code
```python
>>> from itertools import product
>>>
>>> print list(product([1,2,3],repeat = 2))
[(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
>>>
>>> print list(product([1,2,3],[3,4]))
[(1, 3), (1, 4), (2, 3), (2, 4), (3, 3), (3, 4)]
>>>
>>> A = [[1,2,3],[3,4,5]]
>>> print list(product(*A))
[(1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (3, 5)]
>>>
>>> B = [[1,2,3],[3,4,5],[7,8]]
>>> print list(product(*B))
[(1, 3, 7), (1, 3, 8), (1, 4, 7), (1, 4, 8), (1, 5, 7), (1, 5, 8), (2, 3, 7), (2, 3, 8), (2, 4, 7), (2, 4, 8), (2, 5, 7), (2, 5, 8), (3, 3, 7), (3, 3, 8), (3, 4, 7), (3, 4, 8), (3, 5, 7), (3, 5, 8)]
```
- Task
    - You are given a two lists A and B. Your task is to compute their cartesian product A X B.

- Example
```python
A = [1, 2]
B = [3, 4]

AxB = [(1, 3), (1, 4), (2, 3), (2, 4)]
```
- Note: A and B are sorted lists, and the cartesian product's tuples should be output in sorted order.

- Input Format
    - The first line contains the space separated elements of list A.
    - The second line contains the space separated elements of list B.
    - Both lists have no duplicate integer elements.

- Constraints
    - 0 < A < 30
    - 0 < B < 30

- Output Format
    - Output the space separated tuples of the cartesian product.

#### 문제풀이
- list A와 B가 주어졌을때 A X B 의 데카르트의 곱을 구하는 것
- python의 product를 사용하였고, list형태로 받은뒤 해당 list의 원소를 for문으로 print하는 형식으로 만듬
- 아마 다른방법도 있을듯 하다..


```python
from itertools import product
A = list(map(int,(input().split())))
B = list(map(int,(input().split())))
p = list(product(A,B))
for i in p:
    print(i, end=' ')
```

     1 2
     3 4


    (1, 3) (1, 4) (2, 3) (2, 4) 


```python

```
