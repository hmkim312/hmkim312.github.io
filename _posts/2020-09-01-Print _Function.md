---
title: Print Function (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/python-print/problem>{:target="_blank"}


- The included code stub will read an integer, n, from STDIN.
- Without using any string methods, try to print the following:
    - 1,2,3 ... n 
- Note that "..." represents the consecutive values in between.

- Example
    - n = 5
    - Print the string 12345.

- Input Format
    - The first line contains an integer n.

- Constraints
   - 1 <= n <= 150

- Output Format
    - Print the list of integers from 1 through n as a string, without spaces.

#### 문제풀이
- n이 주어지면 해당 1부터n까지 프린트하면됨
- 기본 print 함수의 end = '\n'이 디폴트이기때문에 그냥 이것을 ''띄어쓰기 없는것으로 주면 옆으로 프린트가 됨


```python
if __name__ == '__main__':
    n = int(input())
    for i in range(1, n+1):
        print(i, end='')
```

     5


    12345
