---
title: Iterables and Iterators (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/iterables-and-iterators/problem>{:target="_blank"}

- The itertools module standardizes a core set of fast, memory efficient tools that are useful by themselves or in combination. Together, they form an iterator algebra making it possible to construct specialized tools succinctly and efficiently in pure Python.

- To read more about the functions in this module, check out their documentation here.

- You are given a list of N lowercase English letters. For a given integer K, you can select any K indices (assume 1- based indexing) with a uniform probability from the list.

- Find the probability that at least one of the K indices selected will contain the letter: 'a'.

- Input Format
    - The input consists of three lines. The first line contains the integer N, denoting the length of the list. The next line consists of N space-separated lowercase English letters, denoting the elements of the list.

    - The third and the last line of input contains the integer K, denoting the number of indices to be selected.

- Output Format
    - Output a single line consisting of the probability that at least one of the K indices selected contains the letter:'a'.

- Note: The answer must be correct up to 3 decimal places.

- Constraints
    - 1 <= N <= 10
    - 1 <= K <= N
    - All the letters in the list are lowercase English letters.

#### 문제풀이
- 문자의 길이 n, 문자열 c, 맵핑할 k가 주어지고 a가 포함되어있는 인덱스와, 전체 인덱스의 비율을 구하는것
- 굳이 인덱스로 하지않고 itertools의 combinations 메서드를 활용하여 (a, a)이런식으로 만들고, 해당 컴비네이션에 a가 있으면 count에 +1씩 하게 만듬
- 마지막으로 count / len 을 하면 비율이 나옴


```python
from itertools import combinations

n = input()
c = input().split()
k = int(input())

count = 0
for i in combinations(c, k):
    if 'a' in i:
        count += 1
print(count / len(list(combinations(c, k))))
```

     4
     a a c d
     2


    0.8333333333333334
