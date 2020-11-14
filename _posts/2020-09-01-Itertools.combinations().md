---
title: Itertools.combinations() (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/itertools-combinations/problem>{:target="_blank"}

- This tool returns the  length subsequences of elements from the input iterable.
- Combinations are emitted in lexicographic sorted order. So, if the input iterable is sorted, the combination tuples will be produced in sorted order.
- Sample Code
```python
>>> from itertools import combinations
>>> 
>>> print list(combinations('12345',2))
[('1', '2'), ('1', '3'), ('1', '4'), ('1', '5'), ('2', '3'), ('2', '4'), ('2', '5'), ('3', '4'), ('3', '5'), ('4', '5')]
>>> 
>>> A = [1,1,3,3,3]
>>> print list(combinations(A,4))
[(1, 1, 3, 3), (1, 1, 3, 3), (1, 1, 3, 3), (1, 3, 3, 3), (1, 3, 3, 3)]
```

- Task
    - You are given a string S.
    - Your task is to print all possible combinations, up to size k, of the string in lexicographic sorted order.

- Input Format
    - A single line containing the string S and integer value k separated by a space.

- Constraints
    - 0 < K <= len(s) 
    - The string contains only UPPERCASE characters.

- Output Format
    - Print the different combinations of string S on separate lines.

#### 문제풀이
- itertools의 combinations를 사용하여 주어진 S를 1 ~ k 갯수만큼 순서대로 출력하는것
- 일전에 했던 내용은 주어진 k의 숫자만큼만 출력하는 것이었다면 이번엔 1 ~ k까지임
- for문을 1번더 써서 해결


```python
from itertools import combinations

s, k = input().split()
k = int(k)
s = sorted(s)
for i in range(1, k+1):
    for c in list(combinations(s, i)):
        print(''.join(c))
```

     HACK 2


    A
    C
    H
    K
    AC
    AH
    AK
    CH
    CK
    HK

