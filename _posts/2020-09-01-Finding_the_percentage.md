---
title: Finding the percentage (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/finding-the-percentage/problem>{:target="_blank"}


- The provided code stub will read in a dictionary containing key/value pairs of name:[marks] for a list of students. Print the average of the marks array for the student name provided, showing 2 places after the decimal.

- Example
    - marks key : value pairs are
    - 'alpha' : [20, 30, 40]
    - 'beta' : [30, 50, 70]
    - query_name : 'beta'

- The query_name is 'beta'. beta's average score is (30 + 50 + 70) / 3 = 50.0.

- Input Format
    - The first line contains the integer n, the number of students' records. The next n lines contain the names and marks obtained by a student, each value separated by a space. The final line contains query_name, the name of a student to query.

- Constraints
    - 2 <= n <= 10
    - 0 <= marks[i] <= 100
    - lenth of marks array = 3

- Output Format
    - Print one line: The average of the marks obtained by the particular student correct to 2 decimal places.

#### 문제풀이
- 이름과 숫자가 주어지고, 마지막에 입력된 이름에 대해 숫자에 대한 평균을 구하는것
- sum과 len 함수를 이용하여 평균을 구하였다.


```python
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    print('%0.2f' % (sum(student_marks[query_name])/len(student_marks[query_name])))
```

     2
     Harsh 25 26.5 28
     Anurag 26 28 30
     Harsh


    26.50