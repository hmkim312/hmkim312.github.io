---
title: itertools.combinations with replacement() (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/itertools-combinations-with-replacement/problem>{:target="_blank"}

- This tool returns r length subsequences of elements from the input iterable allowing individual elements to be repeated more than once.
- Combinations are emitted in lexicographic sorted order. So, if the input iterable is sorted, the combination tuples will be produced in sorted order.

- Sample Code
```python
>>> from itertools import combinations_with_replacement
>>> 
>>> print list(combinations_with_replacement('12345',2))
[('1', '1'), ('1', '2'), ('1', '3'), ('1', '4'), ('1', '5'), ('2', '2'), ('2', '3'), ('2', '4'), ('2', '5'), ('3', '3'), ('3', '4'), ('3', '5'), ('4', '4'), ('4', '5'), ('5', '5')]
>>> 
>>> A = [1,1,3,3,3]
>>> print list(combinations(A,2))
[(1, 1), (1, 3), (1, 3), (1, 3), (1, 3), (1, 3), (1, 3), (3, 3), (3, 3), (3, 3)]
```

- Task
    - You are given a string S.
    - Your task is to print all possible size k replacement combinations of the string in lexicographic sorted order.

- Input Format
    - A single line containing the string S and integer value k separated by a space.

- Constraints
    - 0 < k <= len(s)
    - The string contains only UPPERCASE characters.

- Output Format
    - Print the combinations with their replacements of string S on separate lines.

#### 문제 풀이
- 아마 combinations_with_replacement를 사용하는것이고, 앞에서 배운 itertools의 메서드들이 다른것을 설명하는듯 싶다.
- 주어진 문자열 s에대해 k갯수만큼 끊어서 나오는 모든 조합의 수를 생성
- 이번엔 combinations_with_replacement를 사용하는것
- 로직은 앞에 것들과 똑같다.


```python
from itertools import combinations_with_replacement

s ,k = input().split()
s = sorted(s)
k = int(k)

from itertools import combinations_with_replacement
c = list(combinations_with_replacement(s,k))
for i in c:
    print(''.join(i))
```

     HACK 2


    AA
    AC
    AH
    AK
    CC
    CH
    CK
    HH
    HK
    KK