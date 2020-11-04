---
title: Day25 - Running Time and Complexity (Python 3)
author: HyunMin Kim
date: 2020-06-05 00:00:00 0000
categories: [Hacker Ranker, 30 Days of Code]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/30-running-time-and-complexity/problem>{:target="_blank"}

- Objective
    - Today we're learning about running time! Check out the Tutorial tab for learning materials and an instructional video!

- Task
    - A prime is a natural number greater than 1 that has no positive divisors other than 1 and itself. Given a number, n, determine and print whether it's Prime or Not Prime.

- Note: If possible, try to come up with a O(rootN) primality algorithm, or see what sort of optimizations you come up with for an O(n) algorithm. Be sure to check out the Editorial after submitting your code!

- Input Format
    - The first line contains an integer, T, the number of test cases.
    - Each of the T subsequent lines contains an integer, n, to be tested for primality.

- Constraints
    - 1 <= T <= 30
    - 1 <= N <= 2 * 10 ^9
- Output Format
    - For each test case, print whether n is Prime or Not Prime on a new line.

#### 문제풀이
- 주어진 n에 대하여 소수를 찾는 함수
- 소수는 1과 자기자신으로만 나누어 떨어지는 수
- 2는 소수이나, 나머지 짝수는 소수가 아님
- 1은 소수가 아님
- 위의 내용을 종합하여 1은 소수가 아니고 2와 3은 소수이며 짝수는 소수가 아닌 예외처리
- 이후 3부터 n까지 홀수만 나누는것을 코드화하였으나 시간초과
- N의 약수는 무조건 N의 제곱근의 범위에 존재하는 특성으로 제곱근을 활용하여 시간을 줄임


```python
K = None
n = int(input())
```

     33



```python
# 소수를 찾는 함수를 만들어서 제출했으나, 시간 초과
def num_prime(n):
    if n < 2:
        return print('Not prime')
    elif n == 2 or  n == 3:
        return print('Prime')
    if n % 2 == 0:
        return print('Not prime')
    for j in range(3, n, 2):
        if n % j == 0:
            return print('Not prime')
    else :
        return print('Prime')
```


```python
num_prime(67)
```

    Prime



```python
# 소수를 찾는 함수를 만들어서 제출했으나, 소수의 특성중 하나인 제곱근을 활용하여 품
import math

def num_prime(n):
    if n < 2:
        return print('Not prime')
    elif n == 2 or  n == 3:
        return print('Prime')
    if n % 2 == 0:
        return print('Not prime')
    k = int(round(math.sqrt(n))) + 1
    for i in range(3, k, 2): 
        if n % i is 0: 
            return print('Not prime')
    else :
        return print('Prime')
```


```python
num_prime(5)
```

    3
    Prime



```python
# 전체 제출 코드
import math
def num_prime(n):
    if n < 2:
        return print('Not prime')
    elif n == 2 or  n == 3:
        return print('Prime')
    if n % 2 == 0:
        return print('Not prime')
    k = int(round(math.sqrt(n))) + 1
    for i in range(3, k, 2): 
        if n % i is 0: 
            return print('Not prime')
    else :
        return print('Prime')
T = int(input())
for i in range(T):
    n = int(input())
    num_prime(n)
```

     1
     33


    7
    Not prime

