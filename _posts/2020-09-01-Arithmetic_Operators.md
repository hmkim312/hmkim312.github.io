---
title: Arithmetic Operators (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/python-arithmetic-operators/problem>{:target="_blank"}


- Task
    - The provided code stub reads two integers from STDIN,  and . Add code to print three lines where:
        - 1. The first line contains the sum of the two numbers.
        - 2. The second line contains the difference of the two numbers (first - second).
        - 3. The third line contains the product of the two numbers.
- Example
    - a = 3
    - b = 5

- Print the following:
    - 8
    - -2
    - 15
    
- Input Format
    - The first line contains the first integer, a.
    - The second line contains the second integer, b.

- Constraints
    - 1 <= a <= 10^10
    - 1 <= b <= 10^10

- Output Format
    - Print the three lines as explained above.


```python
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a + b)
    print(a - b)
    print(a * b)
```

     3
     5


    8
    -2
    15

