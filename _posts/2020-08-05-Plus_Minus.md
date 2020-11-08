---
title: Plus Minus (Python 3)
author: HyunMin Kim
date: 2020-08-05 00:00:00 0000
categories: [Hacker Ranker, Problem Solving]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/plus-minus/problem>{:target="_blank"}


Given an array of integers, calculate the ratios of its elements that are positive, negative, and zero. Print the decimal value of each fraction on a new line with 6 places after the decimal.

- Note: This challenge introduces precision problems. The test cases are scaled to six decimal places, though answers with absolute error of up to 10^-4 are acceptable.

- Example
    - arr = [1,1,0,-1,-1]
    - There are n = 5 elements, two positive, two negative and one zero. Their ratios are 2/5 = 0.400000, 2/5 = 0.400000 and 1/5 = 0.200000. Results are printed as:

```python
0.400000
0.400000
0.200000
```

- Function Description
    - Complete the plusMinus function in the editor below.
    - plusMinus has the following parameter(s):
        - int arr[n]: an array of integers
- Print
    - Print the ratios of positive, negative and zero values in the array. Each value should be printed on a separate line with  6 digits after the decimal. The function should not return a value.

- Input Format
    - The first line contains an integer, n, the size of the array.
    - The second line contains n space-separated integers that describe arr[n].

- Constraints
    - 0 < n <= 100
    - -100 <= arr[i] <= 100

- Output Format
    - Print the following 3 lines, each to 6 decimals:
    1. proportion of positive values
    2. proportion of negative values
    3. proportion of zeros

#### 문제풀이
- 주어진 arr에서 음수, 양수, 0을 확인하고  전체 갯수를 음수, 양수, 0의 갯수로 나눈 값을 출력하기
- for문을 돌면서 음수면 음수,양수면 양수 등 할수있으나, fillter 함수와 lambda 함수를 통해 양, 음, 0을 더 빠르게 확인.
- fillter 후에 list시키고 그 갯수(len)만큼 나누는 코드를 생성


```python
import math
import os
import random
import re
import sys

# Complete the plusMinus function below.
def plusMinus(arr):
    print(round(len(list(filter(lambda x : x > 0, arr))) / len(arr), 6))
    print(round(len(list(filter(lambda x : x < 0, arr))) / len(arr), 6))
    print(round(len(list(filter(lambda x : x == 0, arr))) / len(arr), 6))

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    plusMinus(arr)
```

     6
     -4 3 -9 0 4 1


    0.5
    0.5
    0.5

