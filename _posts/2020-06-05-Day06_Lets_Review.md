---
title: Day06 - Let's Review (Python 3)
author: HyunMin Kim
date: 2020-06-05 00:00:00 0000
categories: [Hacker Ranker, 30 Days of Code]
tags: [Loop]
---

- URL : <https://www.hackerrank.com/challenges/30-review-loop/problem>{:target="_blank"}

- Objective
    - Today we're expanding our knowledge of Strings and combining it with what we've already learned about loops. Check out the Tutorial tab for learning materials and an instructional video!

- Task
    - Given a string, S, of length N that is indexed from 0 to N-1, print its even-indexed and odd-indexed characters as  space-separated strings on a single line (see the Sample below for more detail).

- Note: 0 is considered to be an even index.

- Input Format
    - The first line contains an integer, T (the number of test cases).
    - Each line i of the T subsequent lines contain a String, S.

- Constraints
    - 1 <= T <= 10
    - 2 <= lenthofs <= 10000

- Output Format
    - For each String Si (where 0 <= j <= T - 1 ), print Si's even-indexed characters, followed by a space, followed by Si's odd-indexed characters.


```python
t = int(input())
for i in range(t):
    s = input()
    a, b= s[::2], s[1::2]
    print(a, b)
```

     3
     hi


    h i


     hyunmin


    humn yni


     sdjkasld


    sjal dksd
