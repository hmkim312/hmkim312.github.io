---
title: Itertools.permutations() (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/itertools-permutations/problem>{:target="_blank"}

- This tool returns successive r length permutations of elements in an iterable.
- If r is not specified or is None, then r defaults to the length of the iterable, and all possible full length permutations are generated.
- Permutations are printed in a lexicographic sorted order. So, if the input iterable is sorted, the permutation tuples will be produced in a sorted order.

- Sample Code

```python
>>> from itertools import permutations
>>> print permutations(['1','2','3'])
<itertools.permutations object at 0x02A45210>
>>> 
>>> print list(permutations(['1','2','3']))
[('1', '2', '3'), ('1', '3', '2'), ('2', '1', '3'), ('2', '3', '1'), ('3', '1', '2'), ('3', '2', '1')]
>>> 
>>> print list(permutations(['1','2','3'],2))
[('1', '2'), ('1', '3'), ('2', '1'), ('2', '3'), ('3', '1'), ('3', '2')]
>>>
>>> print list(permutations('abc',3))
[('a', 'b', 'c'), ('a', 'c', 'b'), ('b', 'a', 'c'), ('b', 'c', 'a'), ('c', 'a', 'b'), ('c', 'b', 'a')]
```

- Task
    - You are given a string S.
    - Your task is to print all possible permutations of size k of the string in lexicographic sorted order.

- Input Format
    - A single line containing the space separated string s and the integer value k.

- Constraints
    - 0 < k <= len(s)
    - The string contains only UPPERCASE characters.

- Output Format
    - Print the permutations of the string S on separate lines.

#### 문제풀이
- 단어s와 반복될 숫자n이 주어지면, 해당 단어를 n개의 조합으로 만들어서 출력하면됨
- n개의 조합은 itertools의 permutations를 사용하면 됨
- for문으로 해당 조합을 하나하나 뽑고, join을 이용하여 합침,
- 대신 s는 정렬이 되어있어야 해서 s를 sorted 시킨것을 출력함


```python
from itertools import permutations
s , n = input().split()

for i in list(permutations(sorted(s),int(n))):
    print(''.join(i))
```

     HACK 2


    AC
    AH
    AK
    CA
    CH
    CK
    HA
    HC
    HK
    KA
    KC
    KH



```python
map(list,(permutations(s,int(i)))
```




    <map at 0x1c97d3d2808>




```python

```
