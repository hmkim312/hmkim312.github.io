---
title: Birthday Cake Candles (Python 3)
author: HyunMin Kim
date: 2020-08-05 00:00:00 0000
categories: [Hacker Ranker, Problem Solving]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/birthday-cake-candles/problem>{:target="_blank"}


- You are in charge of the cake for your niece's birthday and have decided the cake will have one candle for each year of her total age. When she blows out the candles, she’ll only be able to blow out the tallest ones. Your task is to find out how many candles she can successfully blow out.

- For example, if your niece is turning 4 years old, and the cake will have 4 candles of height 4, 4, 1, 3, she will be able to blow out 2 candles successfully, since the tallest candles are of height 4 and there are 2 such candles.

- Function Description
    - Complete the function birthdayCakeCandles in the editor below. It must return an integer representing the number of candles she can blow out.
    - birthdayCakeCandles has the following parameter(s):
        - ar: an array of integers representing candle heights
        
- Input Format
    - The first line contains a single integer, n, denoting the number of candles on the cake.
    - The second line contains n space-separated integers, where each integer i describes the height of candle i.

- Constraints
    - 1 <= n <= 10^5
    - 1 <= ar[i] <= 10^7

- Output Format
    - Return the number of candles that can be blown out on a new line.

#### 문제풀이
- 조카의 나이가 주어지고, 그 나이만큼의 초가 있으며, 조카는 가장 긴 초의 촛불만 끌수 있다.
- 너무 어렵게 생각했음, ar_count는 그냥 초의 갯수인데, 그게 아니라 그거보다 더 큰수가 올수도 있을거라고 생각했음, 문제를 잘 읽어야할듯..
- 어쨋든 배열에서 가장 큰 수를 찾고, 가장 큰 수의 count를 확인하는 birthdayCakeCandles 함수를 작성


```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the birthdayCakeCandles function below.
def birthdayCakeCandles(ar):
    max_candle = max(ar)
    return ar.count(max_candle)
            
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    ar_count = int(input())

    ar = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(ar)

    fptr.write(str(result) + '\n')

    fptr.close()
```
