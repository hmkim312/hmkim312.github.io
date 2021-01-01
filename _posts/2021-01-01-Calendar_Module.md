---
title: Calendar Module (Python 3)
author: HyunMin Kim
date: 2021-01-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- Url : <https://www.hackerrank.com/challenges/calendar-module/problem>{:target="_blank"}

The calendar module allows you to output calendars and provides additional useful functions for them.
class calendar.TextCalendar([firstweekday])
This class can be used to generate plain text calendars.

- Sample Code
```
>>> import calendar
>>> 
>>> print calendar.TextCalendar(firstweekday=6).formatyear(2015)
                                  2015

      January                   February                   March
Su Mo Tu We Th Fr Sa      Su Mo Tu We Th Fr Sa      Su Mo Tu We Th Fr Sa
             1  2  3       1  2  3  4  5  6  7       1  2  3  4  5  6  7
 4  5  6  7  8  9 10       8  9 10 11 12 13 14       8  9 10 11 12 13 14
11 12 13 14 15 16 17      15 16 17 18 19 20 21      15 16 17 18 19 20 21
18 19 20 21 22 23 24      22 23 24 25 26 27 28      22 23 24 25 26 27 28
25 26 27 28 29 30 31                                29 30 31

       April                      May                       June
Su Mo Tu We Th Fr Sa      Su Mo Tu We Th Fr Sa      Su Mo Tu We Th Fr Sa
          1  2  3  4                      1  2          1  2  3  4  5  6
 5  6  7  8  9 10 11       3  4  5  6  7  8  9       7  8  9 10 11 12 13
12 13 14 15 16 17 18      10 11 12 13 14 15 16      14 15 16 17 18 19 20
19 20 21 22 23 24 25      17 18 19 20 21 22 23      21 22 23 24 25 26 27
26 27 28 29 30            24 25 26 27 28 29 30      28 29 30
                          31

        July                     August                  September
Su Mo Tu We Th Fr Sa      Su Mo Tu We Th Fr Sa      Su Mo Tu We Th Fr Sa
          1  2  3  4                         1             1  2  3  4  5
 5  6  7  8  9 10 11       2  3  4  5  6  7  8       6  7  8  9 10 11 12
12 13 14 15 16 17 18       9 10 11 12 13 14 15      13 14 15 16 17 18 19
19 20 21 22 23 24 25      16 17 18 19 20 21 22      20 21 22 23 24 25 26
26 27 28 29 30 31         23 24 25 26 27 28 29      27 28 29 30
                          30 31

      October                   November                  December
Su Mo Tu We Th Fr Sa      Su Mo Tu We Th Fr Sa      Su Mo Tu We Th Fr Sa
             1  2  3       1  2  3  4  5  6  7             1  2  3  4  5
 4  5  6  7  8  9 10       8  9 10 11 12 13 14       6  7  8  9 10 11 12
11 12 13 14 15 16 17      15 16 17 18 19 20 21      13 14 15 16 17 18 19
18 19 20 21 22 23 24      22 23 24 25 26 27 28      20 21 22 23 24 25 26
25 26 27 28 29 30 31      29 30                     27 28 29 30 31
```


- Task
    - You are given a date. Your task is to find what the day is on that date.

- Input Format

    - A single line of input containing the space separated month, day and year, respectively, in MM/DD/YYYY format.

- Constraints
    - 2000 < year < 3000

- Output Format
 - Output the correct day in capital letters.

#### 문제 해설
- MM/DD/YEAR (08 05 2015)로 이루어진 데이터를 입력받으면 해당 날짜에 맞는 요일을 출력하라는것
- 위의 예시에 있는 calendar는 import가 되지 않는다.
- 다른 패키지인 datetime을 import 하여 month, day, year를 저장 후 weekdays라는 리스트에 미리 (월 ~ 일)을 저장한다.(datetime은 0이 월요일)
- datetime.date(year, month, day)를 넣으면 날짜가 생성되고, weekday 메서드로 해당 날짜에 맞는 요일을 숫자 (0 ~ 6)로 리턴받아 weekdays에 맞는 요일을 출력하게 작성하면 끝


```python
import datetime
n = input()
month = int(n[0:2])
day = int(n[3:5])
year = int(n[6:])
weekdays = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']
date = datetime.date(year, month, day)
print(weekdays[date.weekday()])
```

     08 05 2015


    WEDNESDAY

