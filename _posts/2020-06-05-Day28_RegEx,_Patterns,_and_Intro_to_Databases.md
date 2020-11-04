---
title: Day28 - RegEx, Patterns, and Intro to Databases (Python 3)
author: HyunMin Kim
date: 2020-06-05 00:00:00 0000
categories: [Hacker Ranker, 30 Days of Code]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/30-regex-patterns/problem>{:target="_blank"}



- Objective
    - Today, we're working with regular expressions. Check out the Tutorial tab for learning materials and an instructional video!

- Task
    - Consider a database table, Emails, which has the attributes First Name and Email ID. Given N rows of data simulating the Emails table, print an alphabetically-ordered list of people whose email address ends in @gmai.com.

- Input Format
    - The first line contains an integer, N, total number of rows in the table.
    - Each of the N subsequent lines contains 2 space-separated strings denoting a person's first name and email ID, respectively.

- Constraints
    - 2 <= N <= 30 
    - Each of the first names consists of lower case letters [a-z] only.
    - Each of the email IDs consists of lower case letters [a-z],@  and  only.
    - The length of the first name is no longer than 20.
    - The length of the email ID is no longer than 50.

- Output Format
    - Print an alphabetically-ordered list of first names for every user with a gmail account. Each name must be printed on a new line.

#### 문제풀이
- 정규표현식을 사용해서 이메일의 도메인이 gmail.com인 id만 가져와서 출력
- 그냥 emailID에 '@gmail.com' 이 있는지 확인해서 있으면 firstName을 append 시키는것도 가능하지만.. 그냥 정규표현식을 써보긴했다..


```python
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    N = int(input())

    firstNamelist = []
    for N_itr in range(N):
        firstNameEmailID = input().split()

        firstName = firstNameEmailID[0]

        emailID = firstNameEmailID[1]

        emaillist = re.compile('[a-z]+@gmail.com').finditer(emailID)
        for email in emaillist:
            if len(email.group()) >= 1:
                firstNamelist.append(firstName)
    firstNamelist = sorted(firstNamelist)
    for i in firstNamelist:
        print(i)
```

     2
     julia julia@julia.me
     julia sjulia@gmail.com


    julia