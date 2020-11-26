---
title: String Split and Join (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/python-string-split-and-join/problem>{:target="_blank"}

- In Python, a string can be split on a delimiter.

- Example:
```python
>>> a = "this is a string"
>>> a = a.split(" ") # a is converted to a list of strings. 
>>> print a
['this', 'is', 'a', 'string']
```

- Joining a string is simple:

```python
>>> a = "-".join(a)
>>> print a
this-is-a-string 
```
- Task
    - You are given a string. Split the string on a " " (space) delimiter and join using a - hyphen.

- Input Format
    - The first line contains a string consisting of space separated words.

- Output Format
    - Print the formatted string as explained above.

#### 문제풀이
- line 이 주어지면 공백으로 split하여 리스트를 만들고, 그 리스트를 '-'로 join하라는 문제
- 정석으로 풀면 split -> join이 맞고 (1안)
- 그냥 replace(변환함수)로 공백을 '-'로 바꾸면 된다.(2안)


```python
# 1안
def split_and_join(line):
    line = line.split()
    return '-'.join(line)
    
if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)
```




    'this-is-a-string'




```python
# 2안
def split_and_join(line):
    return line.replace(' ', '-')
    
if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)
```
