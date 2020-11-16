---
title: Mutations (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/python-mutations/problem>{:target="_blank"}


- Task
    - Read a given string, change the character at a given index and then print the modified string.

- Input Format
    - The first line contains a string, s.
    - The next line contains an integer i, denoting the index location and a character c separated by a space.

- Output Format
    - Using any of the methods explained above, replace the character at index i with character c.

#### 문제 풀이
- string을 list로 바꾸고 주어진 i번쨰에 있는 원소를 character로 바꾸는 함수를 작성


```python
def mutate_string(string, position, character):
    ls = list(string)
    ls[position] = character
    new_stirng = ''.join(ls)
    return new_stirng
```


```python
string = 'abracadabra'
i = 5
c = 'k'

mutate_string(string, i, c)
```
    'abrackdabra'




```python
def mutate_string(string, position, character):
    ls = list(string)
    ls[position] = character
    new_stirng = ''.join(ls)
    return new_stirng

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)
```
