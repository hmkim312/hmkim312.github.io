---
title: Write a function (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/write-a-function/problem>{:target="_blank"}

- An extra day is added to the calendar almost every four years as February 29, and the day is called a leap day. It corrects the calendar for the fact that our planet takes approximately 365.25 days to orbit the sun. A leap year contains a leap day.

- In the Gregorian calendar, three conditions are used to identify leap years:
    - The year can be evenly divided by 4, is a leap year, unless:
        - The year can be evenly divided by 100, it is NOT a leap year, unless:
            - The year is also evenly divisible by 400. Then it is a leap year.
- This means that in the Gregorian calendar, the years 2000 and 2400 are leap years, while 1800, 1900, 2100, 2200, 2300 and 2500 are NOT leap years. Source

- Task
    - Given a year, determine whether it is a leap year. If it is a leap year, return the Boolean True, otherwise return False.
    
- Note that the code stub provided reads from STDIN and passes arguments to the is_leap function. It is only necessary to complete the is_leap function.

- Input Format
    - Read year, the year to test.

- Constraints
    - 1900 <= year <= 10^5

- Output Format
    - The function must return a Boolean value (True/False). Output is handled by the provided code stub.

#### 문제풀이
- 윤년인지 확인하는 함수 작성
- 윤년은 4년마다 한번씩 오지만, 100년단위로 끊어지면 (ex 1900)은 윤년이 아님.
- 하지만 100년 단위 중에서 400년 마다 1번씩 윤년임
- 아래의 조건들을 순차적으로 하나씩 필터링 하게 하는 함수를 만듬
    - year를 400으로 나누어 0이 되면 윤년
    - year를 100으로 나누어 0이 되면 윤년이 아님
    - year를 4로 나누어 0이되면 윤년
    - 위의 조건이 모두 맞지않으면 윤년이 아님


```python
def is_leap(year):
    leap = False

    # Write your logic here
    if year % 400 == 0:
        leap = True

    elif year % 100 == 0:
        leap = False

    elif year % 4 == 0:
        leap = True

    return leap


year = int(input())
print(is_leap(year))
```

     1994


    False



```python

```
