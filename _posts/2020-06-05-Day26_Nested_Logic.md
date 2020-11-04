---
title: Day26 - Nested Logic (Python 3)
author: HyunMin Kim
date: 2020-06-05 00:00:00 0000
categories: [Hacker Ranker, 30 Days of Code]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/30-nested-logic/problem>{:target="_blank"}

- Objective
    - Today's challenge puts your understanding of nested conditional statements to the test. You already have the knowledge to complete this challenge, but check out the Tutorial tab for a video on testing!

- Task
    - Your local library needs your help! Given the expected and actual return dates for a library book, create a program that calculates the fine (if any). The fee structure is as follows:
        - If the book is returned on or before the expected return date, no fine will be charged (i.e.: fine = 0).
        - If the book is returned after the expected return day but still within the same calendar month and year as the expected return date, fine = 15 Hackos x (the number of days late).
        - If the book is returned after the expected return month but still within the same calendar year as the expected return date, fine = 500 Hackos x (the number of months late) .
        - If the book is returned after the calendar year in which it was expected, there is a fixed fine of 10000Hacoks.

- Input Format
    - The first line contains 3 space-separated integers denoting the respective day, month, and year on which the book was actually returned.
    - The second line contains 3 space-separated integers denoting the respective day, month, and year on which the book was expected to be returned (due date).

- Constraints
    - 1 <= D <= 31
    - 1 <= M <= 12
    - 1 <= Y <= 3000
    - It is guarandteed that dates will be valid Gregorian calendar dates

- Output Format
    - Print a single integer denoting the library fine for the book received as input.

#### 문제풀이
- 년 월 일 을 비교하여 제출기한보다 늦으면 일정금액만큼 벌금을 만들어 내는 코드
- 예상일자보다 먼저 제출하면 벌금은 0
- 그외 연도가 넘어가면 10000, 월이 넘어가면 넘어간 월 * 500, 날짜만 넘어가면 넘어간 날짜 * 15
- 위의 순서로 코드를 작성


```python
# Enter your code here. Read input from STDIN. Print output to STDOUT
import datetime
n = input()
t = input()
n = n.split()
t = t.split()
dt_n = datetime.datetime(int(n[2]), int(n[1]), int(n[0]))
dt_t = datetime.datetime(int(t[2]), int(t[1]), int(t[0]))
result = 0

if dt_n < dt_t:
    result += 0
elif int(n[2]) > int(t[2]):
    result += 10000
elif int(n[1]) > int(t[1]):
    result += (int(n[1]) - int(t[1])) * 500
elif int(n[0]) > int(t[0]):
    result += (int(n[0]) - int(t[0])) * 15
print(result)
```

     9 6 2015
     6 6 2015


    45
