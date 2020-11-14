---
title: List Comprehensions (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/list-comprehensions/problem>{:target="_blank"}

Let's learn about list comprehensions! You are given three integers x, y and z representing the dimensions of a cuboid along with an integer n. Print a list of all possible coordinates given by (i,j,k) on a 3D grid where the sum of i + j + k is not equal to n. Here, 0 <= i <= x; 0 <= j <= y; 0 <= k <= z;. Please use list comprehensions rather than multiple loops, as a learning exercise.

- Example
    - x = 1
    - y = 1
    - z = 2
    - n = 3

All permutations of [i,j,k] are:
[[0,0,0],[0,0,1],[0,0,2],[0,1,0],[0,1,1],[0,1,2],[1,0,0],[1,0,1],[1,0,2],[1,1,0],[1,1,1],[1,1,2]].

Print an array of the elements that do not sum to n = 3.
[[0,0,0],[0,0,1],[0,0,2],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,2]].

- Input Format
    - Four integers  x, y, z and n, each on a separate line.

- Constraints
    - Print the list in lexicographic increasing order.

#### 문제풀이
- 무언가 list comprehensions으로 풀어야할것 같지만.. 배움이 아직 짧은듯 싶다.
- i,j,k는 x,y,z보다 숫자가 작거나 같고, 그 원소들로 [i,j,k]의 리스트를 만들면 되니, for문으로 각각 append 시켜주는 코드를 생성하였다.


```python
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
```

     1
     1
     2
     3



```python
total_ls = []
ls = []
for i in range(x+1):
    for j in range(y+1):
        for k in range(z+1):
            ls.append(i)
            ls.append(j)
            ls.append(k)
            if sum(ls) != n:
                total_ls.append(ls)
            ls = []
```


```python
total_ls
```




    [[0, 0, 0],
     [0, 0, 1],
     [0, 0, 2],
     [0, 1, 0],
     [0, 1, 1],
     [1, 0, 0],
     [1, 0, 1],
     [1, 1, 0],
     [1, 1, 2]]




```python

```
