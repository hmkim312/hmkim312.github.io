---
title: Mini-Max Sum (Python 3)
author: HyunMin Kim
date: 2020-08-05 00:00:00 0000
categories: [Hacker Ranker, Problem Solving]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/mini-max-sum/problem>{:target="_blank"}

- Given five positive integers, find the minimum and maximum values that can be calculated by summing exactly four of the five integers. Then print the respective minimum and maximum values as a single line of two space-separated long integers.

- Example
    - arr = [1,3,5,7,9]
    - The minimum sum is 1 + 3 + 5 + 7 = 16  and the maximum sum is 3 + 5 + 7 + 9 = 24. The function prints
```python
16 24
```

- Function Description
    - Complete the miniMaxSum function in the editor below.
    - miniMaxSum has the following parameter(s):
    - arr: an array of 5 integers
    
- Print
    - Print two space-separated integers on one line: the minimum sum and the maximum sum of 4 of 5 elements.

- Input Format
    - A single line of five space-separated integers.

- Constraints
    - 1 <= arr[i] <= 10^9

- Output Format
    - Print two space-separated long integers denoting the respective minimum and maximum values that can be calculated by summing exactly four of the five integers. (The output can be greater than a 32 bit integer.)

#### 문제풀이
- 5개의 원소로 이루어진 배열 arr이 주어졌을때 4개를 더해서 가장 작은수와 가장 큰수를 구하는것
- 그냥 5개를 모두 더한뒤 가장 큰 수는 배열 중 가장 작은 수를 뺀것이고 가장 작은 수는 배열중 가장 큰수를 빼면됨
- 위의 내용을 코드로 작성하여 return 시키는 함수를 작성


```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the miniMaxSum function below.
def miniMaxSum(arr):
    total_arr = sum(arr)
    arr = sorted(arr)
    max_arr = total_arr - arr[0]
    min_arr = total_arr - arr[4]
    return print(min_arr, max_arr)

if __name__ == '__main__':
    arr = list(map(int, input().rstrip().split()))

    miniMaxSum(arr)
```

     1 2 3 4 5


    10 14