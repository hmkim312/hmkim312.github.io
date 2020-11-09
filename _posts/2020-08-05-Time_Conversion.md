---
title: Time Conversion (Python 3)
author: HyunMin Kim
date: 2020-08-05 00:00:00 0000
categories: [Hacker Ranker, Problem Solving]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/time-conversion/problem>{:target="_blank"}


- Given a time in 12-hour AM/PM format, convert it to military (24-hour) time.
- Note: Midnight is 12:00:00AM on a 12-hour clock, and 00:00:00 on a 24-hour clock. Noon is 12:00:00PM on a 12-hour clock, and 12:00:00 on a 24-hour clock.

- Function Description
    - Complete the timeConversion function in the editor below. It should return a new string representing the input time in 24 hour format.

- timeConversion has the following parameter(s):
    - s: a string representing time in 12 hour format
    
- Input Format
    - A single string s containing a time in 12-hour clock format (i.e.: hh:mm:ssAM or hh:mm:ssPM), where 01 <= hh <= 12 and 00<= mm,ss <= 59.

- Constraints
    - All input times are valid
    
- Output Format
    - Convert and print the given time in 24-hour format, where 00 <= hh <= 23.

#### 문제 풀이
- 12시간 형식 hh:mm:ssAM or hh:mm:ssPM 으로 되어있는 날짜 타입을 24시간 형식으로 바꾸는 함수 작성
- 파이썬에는 datetime 모듈로 해결하면됨
- 12시간 형식이 저장된 s를 받고, datetime 형식으로 변경 뒤, H:M:S 형식을 리턴


```python
#!/bin/python3

import os
import sys
from datetime import *
#
# Complete the timeConversion function below.
#
def timeConversion(s):
    #
    # Write your code here.
    #
    m2 = datetime.strptime(s, '%I:%M:%S%p')
    return m2.strftime('%H:%M:%S')
         

if __name__ == '__main__':
    f = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = timeConversion(s)

    f.write(result + '\n')

    f.close()
```