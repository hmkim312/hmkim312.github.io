---
title: What's Your Name (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/whats-your-name/problem>{:target="_blank"}

- You are given the firstname and lastname of a person on two different lines. Your task is to read them and print the following:
```python
`Hello firstname lastname! You just delved into python.`
```

- Input Format
    - The first line contains the first name, and the second line contains the last name.

- Constraints
    - The length of the first and last name ≤ 10.

- Output Format
    - Print the output as mentioned above.

#### 문제풀이
- first_name과 last_name을 입력받아, print하는 문제
- f.formatting을 사용하여 해결 끗


```python
def print_full_name(a, b):
    print(f'Hello {a} {b}! You just delved into python.')

if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)
```
