---
title: Capitalize! (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/capitalize/problem>{:target="_blank"}

- You are asked to ensure that the first and last names of people begin with a capital letter in their passports. For example, alison heck should be capitalised correctly as Alison Heck.
    - alison heck => Alison Heck
    - Given a full name, your task is to capitalize the name appropriately.

- Input Format
    - A single line of input containing the full name, S.

- Constraints
    - 0 < len(S) < 1000
    - The string consists of alphanumeric characters and spaces.
- Note: in a word only the first character is capitalized. Example 12abc when capitalized remains 12abc.

- Output Format
    - Print the capitalized string, .

#### 문제 풀이
- 주어진 s에 대해 공백뒤에 있는 소문자를 대문자로 바꾸기
- 주어진 s를 split 함수를 통해 공백(' ')으로 나눈뒤, for문을 통해 해당 문자를 하나하나 앞에만 대문자로 만드는 capitalize()함수를 사용 한뒤 빈 리스트에 넣는다
- 사실 s.title()을 쓰면 되기는하는데, 이거는 앞에 숫자나 특수문자가 있으면 그 뒤에있는 소문자가 대문자로 변경되기 때문에 이번 문제에선 사용하기 힘들었음


```python
def solve(s):
    ls = []
    k = s.split(' ')
    for i in k:
        ls.append(i.capitalize())
    return ' '.join(ls)
```


```python
s = 'hello world'
solve(s)
```




    'Hello World'
