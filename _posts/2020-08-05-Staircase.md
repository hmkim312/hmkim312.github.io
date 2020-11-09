---
title: Staircase (Python 3)
author: HyunMin Kim
date: 2020-08-05 00:00:00 0000
categories: [Hacker Ranker, Problem Solving]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/staircase/problem>{:target="_blank"}


- Consider a staircase of size n = 4:
```
   #
  ##
 ###
####
```

- Observe that its base and height are both equal to n, and the image is drawn using # symbols and spaces. The last line is not preceded by any spaces.
- Write a program that prints a staircase of size n.

- Function Description
    - Complete the staircase function in the editor below. It should print a staircase as described above.
    - staircase has the following parameter(s):
    - n: an integer

- Input Format
    - A single integer, n, denoting the size of the staircase.

- Constraints
    - 0 <= n <= 100

- Output Format
    - Print a staircase of size n using # symbols and spaces.

- Note: The last line must have 0 spaces in it.

#### 문제풀이
- n이 주어지면 오른쪽 정렬이 된 '#' 모양의 계단을 만드는 문제
- (n - i)는 ' '공백으로, i는 '#'로 출력하면됨
- 주의할점은 print할때 sep =''을 주어야 한다는것


```python
def staircase(n):
    for i in range(1, n+1):
        print((n - i) * ' ', (i) * '#', sep = '')
if __name__ == '__main__':
    n = int(input())

    staircase(n)
```

     6


         #
        ##
       ###
      ####
     #####
    ######