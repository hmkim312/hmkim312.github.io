---
title: Compress the String! (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/compress-the-string/problem>{:target="_blank"}


- In this task, we would like for you to appreciate the usefulness of the groupby() function of itertools . To read more about this function, Check this out .

- You are given a string S. Suppose a character 'c' occurs consecutively X times in the string. Replace these consecutive occurrences of the character 'c' with (X, c) in the string.

- For a better understanding of the problem, check the explanation.

- Input Format
    - A single line of input consisting of the string S.

- Output Format
    - A single line of output consisting of the modified string.

- Constraints
    - All the characters of S denote integers between 0 and 9.

#### 문제풀이
- intertools의 groupby 메서드를 사용하여 주어진 문자열 S에 대해 연속된 숫자의 갯수와 해당 숫자를 튜플형식으로 출력하는 것
- print문에서 list를 출력할때 앞에 *를 넣으면 리스트가 벗겨진 채로 출력되는것을 학습함


```python
from itertools import groupby

groups = []
data = input()
for k, g in groupby(data):
    groups.append((len(list(g)), int(k)))
print(*groups)
```

     1223331


    (1, 1) (2, 2) (3, 3) (1, 1)