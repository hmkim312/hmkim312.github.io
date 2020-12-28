---
title: Mod Divmod (Python 3)
author: HyunMin Kim
date: 2020-12-27 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/python-mod-divmod/problem>{:target="_blank"}

- One of the built-in functions of Python is divmod, which takes two arguments  and  and returns a tuple containing the quotient of  first and then the remainder .

- For example:

```
>>> print divmod(177,10)
(17, 7)
```

- Here, the integer division is 177/10 => 17 and the modulo operator is 177%10 => 7.


- Task
    - Read in two integers,  a and b, and print three lines.
    - The first line is the integer division  a//b (While using Python2 remember to import division from __future__).
    - The second line is the result of the modulo operator: a%b.
    - The third line prints the divmod of  a and b.

- Input Format
    - The first line contains the first integer, a, and the second line contains the second integer, b.

- Output Format
    - Print the result as described above.

#### 문제 해설
- python 내장 함수중에 divmod에 대한 이야기입니다.
- a, b가 입력되면 
- 첫번째 줄엔 a//b 즉, a 나누기 b의 몫을 출력 (177 / 10 -> 17)
- 두번째 줄엔 a%b 즉, a 나누기 b의 나머지를 출력 (177 / 10 -> 7)
- 세번째 줄엔 divmod(a, b)를 하여 (17, 7)을 출력 입니다.
- 아주 쉬운 문제 입니다.


```python
a = int(input())
b = int(input())
print(a//b)
print(a % b)
print(divmod(a, b))
```

     177
     10


    17
    7
    (17, 7)



```python

x**3 + x**2 + x + 1 == y
```




    True




```python
x = 1
y = 4
```


```python
f = x**3 + x**2 + x + 1
```


```python
f
```




    4




```python
eval
```




    <function eval(source, globals=None, locals=None, /)>




```python

```
