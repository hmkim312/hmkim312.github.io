---
title: Day19 - Interfaces (Python 3)
author: HyunMin Kim
date: 2020-06-05 00:00:00 0000
categories: [Hacker Ranker, 30 Days of Code]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/30-interfaces/problem>{:target="_blank"}

- Objective
    - Today, we're learning about Interfaces. Check out the Tutorial tab for learning materials and an instructional video!

- Task
    - The AdvancedArithmetic interface and the method declaration for the abstract divisorSum(n) method are provided for you in the editor below.
    - Complete the implementation of Calculator class, which implements the AdvancedArithmetic interface. The implementation for the divisorSum(n) method must return the sum of all divisors of n.

- Input Format
    - A single line containing an integer, n.

- Constraints
    - 1 <= n <= 1000

- Output Format
    - You are not responsible for printing anything to stdout. The locked template code in the editor below will call your code and print the necessary output.

#### 문제풀이
- 주어진 정수 n의 약수를 모두 더하면 된다.
- 주어진 수 만큼 for문을 돌고 나누어서 0이되는 i만 더하는 함수를 작성


```python
class AdvancedArithmetic(object):
    def divisorSum(n):
        raise NotImplementedError

    def divisorSum(self, n):
        num = 0
        for i in range(1, n+1):
            if n % i == 0:
                num += i
        return num
```