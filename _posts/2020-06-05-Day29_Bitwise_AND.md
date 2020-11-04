---
title: Day29 - Bitwise AND (Python 3)
author: HyunMin Kim
date: 2020-06-05 00:00:00 0000
categories: [Hacker Ranker, 30 Days of Code]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/30-bitwise-and/problem>{:target="_blank"}

- Objective
    - Welcome to the last day! Today, we're discussing bitwise operations. Check out the Tutorial tab for learning materials and an instructional video!

- Task
    - Given set S = {1,2,3,4...N}. Find two integers, A and B (where A < B), from set S such that the value of A & B is the maximum possible and also less than a given integer, K. In this case, & represents the bitwise AND operator.

- Input Format
    - The first line contains an integer, T, the number of test cases.
    - Each of the T subsequent lines defines a test case as 2 space-separated integers,  and , respectively.

- Constraints
    - 1 <= T <= 10^3
    - 2 <= N <= 10^3
    - 2 <= K <= N

- Output Format
    - For each test case, print the maximum possible value of A & B on a new line.

#### 문제풀이
- 비트연산자가 무엇인지 처음에 잘 몰라서 헤매였으나, 비트연산자에 대해 개념 이해만 하고 코드를 제출하였으나, 시간 제한에 걸림
- 구글링해서 k가 홀수일때와 짝수일때의 비트연산자 or(|)를 사용해서 제출하는 방법을 알아냄. 정확히 이해는 안가는 부분이라 따로 학습이 필요할듯


```python
n = 2
k = 2
```


```python
# 구글링해서 k가 홀수일때와 짝수일때의 비트연산자 or(|)를 사용해서 제출하는 방법을 알아냄. 
# 정확히 이해는 안가는 부분이라 따로 학습이 필요할듯
if (k|(k-1)) <= n:
    print(k-1)
else:
    print(k-2)
```

    0



```python
# 시간 제한에 걸림 for가 2개라서..
maximum = 0
for a in range(n):
    for b in range(a+1, n+1):
        if k > (a & b) > maximum:
            maximum = a & b
print(maximum)
```

    4

