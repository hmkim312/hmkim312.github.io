---
title: Designer Door Mat (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/designer-door-mat/problem>{:target="_blank"}

- Mr. Vincent works in a door mat manufacturing company. One day, he designed a new door mat with the following specifications:

    - Mat size must be N X M. (N is an odd natural number, and M is 3 times N.)
    - The design should have 'WELCOME' written in the center.
    - The design pattern should only use |, . and - characters.
- Sample Designs
```
    Size: 7 x 21 
    ---------.|.---------
    ------.|..|..|.------
    ---.|..|..|..|..|.---
    -------WELCOME-------
    ---.|..|..|..|..|.---
    ------.|..|..|.------
    ---------.|.---------
    
    Size: 11 x 33
    ---------------.|.---------------
    ------------.|..|..|.------------
    ---------.|..|..|..|..|.---------
    ------.|..|..|..|..|..|..|.------
    ---.|..|..|..|..|..|..|..|..|.---
    -------------WELCOME-------------
    ---.|..|..|..|..|..|..|..|..|.---
    ------.|..|..|..|..|..|..|.------
    ---------.|..|..|..|..|.---------
    ------------.|..|..|.------------
    ---------------.|.---------------
```
- Input Format
    - A single line containing the space separated values of N and M.

- Constraints
    - 5 < N < 101
    - 15 < M < 303

- Output Format
    - Output the design pattern.

#### 문제풀이
- n과 m이 주어지면 해당 숫자에 맞는 패턴의 그림을 그리는것
- 어제 배운 str.center를 이용하여 그림
- 정중앙에는 WELCOME이라는 단어가 들어가므로, for문을 정중앙을 기점으로 데칼코마니처럼 그리게 만듬
- j라는 가중치를 두어 for문이 1번 돌때마다 +1 혹은 -1 씩 하게 만들어 패턴 모양의 횟수를 만들어냄


```python
n = 7
m = 21
k = n // 2
c = '.|.'
j = 0

for i in range(1, k + 1):
    print((c * (i + j)).center(m, '-'))
    j += 1
    
print('WELCOME'.center(m, '-'))

j -= 1
for i in range(k):
    print((c * (k - i + j)).center(m, '-'))
    j -= 1
```

    ---------.|.---------
    ------.|..|..|.------
    ---.|..|..|..|..|.---
    -------WELCOME-------
    ---.|..|..|..|..|.---
    ------.|..|..|.------
    ---------.|.---------
