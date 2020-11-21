---
title: Python_Division (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/python-division/problem>{:target="_blank"}

- Task
    - The provided code stub reads two integers, a and b, from STDIN.

    - Add logic to print two lines. The first line should contain the result of integer division, a // b. The second line should contain the result of float division, a / b.

    - No rounding or formatting is necessary.

- Example
    - a = 3
    - b = 5

    - The result of the integer division 3//5 = 0.
    - The result of the float division is 3/5 = 0.6.
    
- Print:
    - 0
    - 0.6
    
- Input Format
    - The first line contains the first integer, a.
    - The second line contains the second integer, b.

- Output Format
    - Print the two lines as described above.

```python
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
    print(a//b)
    print(a/b)
```