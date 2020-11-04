---
title: Day10 - Binary Numbers (Python 3)
author: HyunMin Kim
date: 2020-06-05 00:00:00 0000
categories: [Hacker Ranker, 30 Days of Code]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/30-binary-numbers/problem>{:target="_blank"}

- Objective
    - Today, we're working with binary numbers. Check out the Tutorial tab for learning materials and an instructional video!

- Task
    - Given a base-10 integer, n, convert it to binary (base-2). Then find and print the base-10 integer denoting the maximum number of consecutive 1's in n's binary representation.

- Input Format
    - A single integer, n.

- Constraints
    - 1 <= n <= 10
    
- Output Format
    - Print a single base-10 integer denoting the maximum number of consecutive 1's in the binary representation of n.

#### 문제 풀이
- 주어진 10진법 정수 n을 2진법으로 바꾸고, 연속된 1을 찾는것
- 입력된 n을 2진법으로 바꿈 -> 0을 기준으로 분리 -> 내림차순으로 정렬 -> 맨 앞의 1의 숫자 갯수를 셈


```python
import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input())
    binary = format(n,'b')
    print(len(sorted(binary.split('0'), reverse=True)[0]))
```

     13
    2