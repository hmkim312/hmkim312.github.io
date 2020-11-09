---
title: Alphabet Rangoli (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/alphabet-rangoli/problem>{:target="_blank"}

- You are given an integer, N. Your task is to print an alphabet rangoli of size N. (Rangoli is a form of Indian folk art based on creation of patterns.)

- Different sizes of alphabet rangoli are shown below:

```python
#size 3

----c----
--c-b-c--
c-b-a-b-c
--c-b-c--
----c----

#size 5

--------e--------
------e-d-e------
----e-d-c-d-e----
--e-d-c-b-c-d-e--
e-d-c-b-a-b-c-d-e
--e-d-c-b-c-d-e--
----e-d-c-d-e----
------e-d-e------
--------e--------

#size 10

------------------j------------------
----------------j-i-j----------------
--------------j-i-h-i-j--------------
------------j-i-h-g-h-i-j------------
----------j-i-h-g-f-g-h-i-j----------
--------j-i-h-g-f-e-f-g-h-i-j--------
------j-i-h-g-f-e-d-e-f-g-h-i-j------
----j-i-h-g-f-e-d-c-d-e-f-g-h-i-j----
--j-i-h-g-f-e-d-c-b-c-d-e-f-g-h-i-j--
j-i-h-g-f-e-d-c-b-a-b-c-d-e-f-g-h-i-j
--j-i-h-g-f-e-d-c-b-c-d-e-f-g-h-i-j--
----j-i-h-g-f-e-d-c-d-e-f-g-h-i-j----
------j-i-h-g-f-e-d-e-f-g-h-i-j------
--------j-i-h-g-f-e-f-g-h-i-j--------
----------j-i-h-g-f-g-h-i-j----------
------------j-i-h-g-h-i-j------------
--------------j-i-h-i-j--------------
----------------j-i-j----------------
------------------j------------------
```
The center of the rangoli has the first alphabet letter a, and the boundary has the Nth alphabet letter (in alphabetical order).

- Input Format
    - Only one line of input containing N, the size of the rangoli.

- Constraints
    - 0 < N < 27

- Output Format
    - Print the alphabet rangoli in the format explained above.

#### 문제풀이
- 숫자 n이 주어지면 1 ~ n 까지의 알파벳을 위의 그림처럼 출력하는 함수를 작성
- 알파벳을 a~z까지 저장(1 ~ 26)
- 일단 range를역순으로 (n-1, -n, -1)의 순서를 주어 출력하고 (ex: n = 3이면 (2,1,0,-1,-2)이 출력) 알파벳을 인덱싱 및 join으로 묶음
- 이후 center 함수를 사용하여 가로(너비)만큼 구해주면됨.
- 참고로 가로구하는 공식은  세로  * 2 - 1 이고 세로는  n * 2 - 1 이다. 이를 풀어 쓰면 (n * 2 - 1) * 2 - 1이고 이룰 전개하면 4 * n - 3이 된다


```python
def print_rangoli(n):
    alpha = 'abcdefghijklmnopqrstuvwxyz'
    for i in range(n-1, -n, -1):
        string = '-'.join(alpha[n-1:abs(i):-1] + alpha[abs(i):n])
        print(string.center(4 * n - 3, '-'))
```


```python
print_rangoli(10)
```

    ------------------j------------------
    ----------------j-i-j----------------
    --------------j-i-h-i-j--------------
    ------------j-i-h-g-h-i-j------------
    ----------j-i-h-g-f-g-h-i-j----------
    --------j-i-h-g-f-e-f-g-h-i-j--------
    ------j-i-h-g-f-e-d-e-f-g-h-i-j------
    ----j-i-h-g-f-e-d-c-d-e-f-g-h-i-j----
    --j-i-h-g-f-e-d-c-b-c-d-e-f-g-h-i-j--
    j-i-h-g-f-e-d-c-b-a-b-c-d-e-f-g-h-i-j
    --j-i-h-g-f-e-d-c-b-c-d-e-f-g-h-i-j--
    ----j-i-h-g-f-e-d-c-d-e-f-g-h-i-j----
    ------j-i-h-g-f-e-d-e-f-g-h-i-j------
    --------j-i-h-g-f-e-f-g-h-i-j--------
    ----------j-i-h-g-f-g-h-i-j----------
    ------------j-i-h-g-h-i-j------------
    --------------j-i-h-i-j--------------
    ----------------j-i-j----------------
    ------------------j------------------
