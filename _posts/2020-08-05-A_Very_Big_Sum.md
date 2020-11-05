---
title: A Very Big Sum (Python 3)
author: HyunMin Kim
date: 2020-08-05 00:00:00 0000
categories: [Hacker Ranker, Problem Solving]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/a-very-big-sum/problem>{:target="_blank"}

- In this challenge, you are required to calculate and print the sum of the elements in an array, keeping in mind that some of those integers may be quite large.

- Function Description
    - Complete the aVeryBigSum function in the editor below. It must return the sum of all array elements.
    - aVeryBigSum has the following parameter(s):
        - int ar[n]: an array of integers.
        
- Return
    - int: the sum of all array elements

- Input Format
    - The first line of the input consists of an integer n.
    - The next line contains n space-separated integers contained in the array.

- Output Format
    - Return the integer sum of the elements in the array.

- Constraints
    - 1 <= n <= 10
    - 0 <= ar[i] <= 10^10

#### 문제풀이
- arr가 주어지고, 그것의 합을 구하라는 문제
- 그냥 sum함수 쓰면됨


```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the aVeryBigSum function below.
def aVeryBigSum(ar):
    return(sum(ar))
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    ar_count = int(input())

    ar = list(map(int, input().rstrip().split()))

    result = aVeryBigSum(ar)

    fptr.write(str(result) + '\n')

    fptr.close()
```