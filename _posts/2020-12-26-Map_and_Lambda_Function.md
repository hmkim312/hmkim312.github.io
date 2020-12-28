---
title: Map and Lambda Function (Python 3)
author: HyunMin Kim
date: 2020-12-26 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/map-and-lambda-expression/problem>{target="_blank"}

- Let's learn some new Python concepts! You have to generate a list of the first N fibonacci numbers, 0 being the first number. Then, apply the map function and a lambda expression to cube each fibonacci number and print the list.

- Concept
    - The map() function applies a function to every member of an iterable and returns the result. It takes two parameters: first, the function that is to be applied and secondly, the iterables.
    - Let's say you are given a list of names, and you have to print a list that contains the length of each name.

```
>> print (list(map(len, ['Tina', 'Raj', 'Tom'])))  
[4, 3, 3]  
Lambda is a single expression anonymous function often used as an inline function. In simple words, it is a function that has only one line in its body. It proves very handy in functional and GUI programming.

>> sum = lambda a, b, c: a + b + c
>> sum(1, 2, 3)
6
```

- Note:

    - Lambda functions cannot use the return statement and can only have a single expression. Unlike def, which creates a function and assigns it a name, lambda creates a function and returns the function itself. Lambda can be used inside lists and dictionaries.

- Input Format
    - One line of input: an integer N.

- Constraints
    - 0 <= N  <= 15
- Output Format
    - A list on a single line containing the cubes of the first N fibonacci numbers.

#### 일반적 문제풀이
- integer N이 주어지면 해당 N의 피보나치수열을 구하고 그 수열들을 3제곱 하는것
- lamda function을 이용해 fibo 함수를 만들고 그 함수를 다시 3제곱하는 lambda 함수를 넣음


```python
n = 5
fibo = lambda n : n if n < 2 else fibo(n-1) + fibo(n-2)
list(map(lambda x : x**3, list(map(fibo, range(n)))))
```




    [0, 1, 1, 8, 27]



#### hackerrank 문제풀이
- cube라는 3제곱하는 함수를 lambda를 이용해 생성
- fibonacci 함수를 생성 하고, ls에 결과를 넣음


```python
cube = lambda x: x ** 3

def fibonacci(n):
    ls = [0, 1]
    for i in range(2,n):
        ls.append(ls[i-1] + ls[i-2])
    return ls[:n]
```


```python
print(list(map(cube, fibonacci(n))))
```

    [0, 1, 1, 8, 27]


- hackerrank 문제풀이
