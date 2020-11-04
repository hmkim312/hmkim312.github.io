---
title: Day17 - More Exceptions (Python 3)
author: HyunMin Kim
date: 2020-06-05 00:00:00 0000
categories: [Hacker Ranker, 30 Days of Code]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/30-more-exceptions/problem>{:target="_blank"}

- Objective
    - Yesterday's challenge taught you to manage exceptional situations by using try and catch blocks. In today's challenge, you're going to practice throwing and propagating an exception. Check out the Tutorial tab for learning materials and an instructional video!

- Task
    - Write a Calculator class with a single method: int power(int,int). The power method takes two integers, n and p, as parameters and returns the integer result of n^p. If either n or p is negative, then the method must throw an exception with the message: n and p should be non-negative.

- Note: Do not use an access modifier (e.g.: public) in the declaration for your Calculator class.

- Input Format
    - Input from stdin is handled for you by the locked stub code in your editor. The first line contains an integer, T, the number of test cases. Each of the T subsequent lines describes a test case in 2 space-separated integers denoting n and p, respectively.

- Constraints
    - No Test Case will result in overflow for correctly written code.
    
- Output Format
    - Output to stdout is handled for you by the locked stub code in your editor. There are T lines of output, where each line contains the result of n^p as calculated by your Calculator class' power method.

#### 문제풀이
- n과 p가 주어지고, n과 p가 모두 0보다 크면(둘중에 하나가 0보다 작다면 false) n ** p를 리턴
- false라면 에러를 생성 (raise Exception("n and p should be non-negative"))


```python
class Calculator:
    def power(self, n, p):
        if n >= 0 and p >= 0:
            return n ** p
        else:
            raise Exception("n and p should be non-negative")
```


```python
myCalculator=Calculator()
T=int(input())
for i in range(T):
    n,p = map(int, input().split())
    try:
        ans=myCalculator.power(n,p)
        print(ans)
    except Exception as e:
        print(e)   
```

     4
     10 0


    1


     0 10


    0


     -1 -3


    n and p should be non-negative


     1 -3


    n and p should be non-negative