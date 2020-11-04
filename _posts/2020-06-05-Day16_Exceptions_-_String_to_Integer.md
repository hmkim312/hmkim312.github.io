---
title: Day16 - Exceptions, String to Integer (Python 3)
author: HyunMin Kim
date: 2020-06-05 00:00:00 0000
categories: [Hacker Ranker, 30 Days of Code]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/30-exceptions-string-to-integer/problem>{:target="_blank"}

- Objective
    - Today, we're getting started with Exceptions by learning how to parse an integer from a string and print a custom error message. Check out the Tutorial tab for learning materials and an instructional video!

- Task
    - Read a string, S, and print its integer value; if S cannot be converted to an integer, print Bad String.

- Note: You must use the String-to-Integer and exception handling constructs built into your submission language. If you attempt to use loops/conditional statements, you will get a 0 score.

- Input Format
    - A single string, S.

- Constraints
    - 1 <= |S| <= 6, where |S| is the length of string .
    - S is composed of either lowercase letters (a-z) or decimal digits (0-9).
- Output Format
    - Print the parsed integer value of S, or Bad String if S cannot be converted to an integer.

#### 문제풀이
- 입력되는 S가 문자형이면 Bad String, 숫자형이면 숫자를 출력하는 내용
- int형으로 변환되면 그대로 출력, 안되면 Bad String을 출력하는 코드를 짬


```python
#!/bin/python3

import sys


S = input().strip()
try :
    print(int(S))
except:
    print('Bad String')
```

    3
    3


