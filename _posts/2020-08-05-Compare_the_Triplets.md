---
title: Compare the Triplets (Python 3)
author: HyunMin Kim
date: 2020-08-05 00:00:00 0000
categories: [Hacker Ranker, Problem Solving]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/compare-the-triplets/problem>{:target="_blank"}

- Alice and Bob each created one problem for HackerRank. A reviewer rates the two challenges, awarding points on a scale from 1 to 100 for three categories: problem clarity, originality, and difficulty.
- The rating for Alice's challenge is the triplet a = (a[0], a[1], a[2]), and the rating for Bob's challenge is the triplet b = (b[0], b[1], b[2]).

- The task is to find their comparison points by comparing a[0] with b[0], a[1] with b[1], and a[2] with b[2].
    - If a[i] > b[i], then Alice is awarded 1 point.
    - If a[i] < b[i], then Bob is awarded 1 point.
    - If a[i] = b[i], then neither person receives a point.
- Comparison points is the total points a person earned.
- Given a and b, determine their respective comparison points.

- Example
   - a = [1, 2, 3]
   - b = [3, 2, 1]

    - For elements *0*, Bob is awarded a point because a[0] .
    - For the equal elements a[1] and b[1], no points are earned.
    - Finally, for elements 2, a[2] > b[2] so Alice receives a point.
- The return array is [1, 1] with Alice's score first and Bob's second.

- Function Description
    - Complete the function compareTriplets in the editor below.
    - compareTriplets has the following parameter(s):
        - int a[3]: Alice's challenge rating
        - int b[3]: Bob's challenge rating
- Return
    - int[2]: Alice's score is in the first position, and Bob's score is in the second.
    
- Input Format
    - The first line contains 3 space-separated integers, a[0], a[1], and a[2], the respective values in triplet a.
    - The second line contains 3 space-separated integers, b[0], b[1], and b[2], the respective values in triplet b.

- Constraints
    - 1 ≤ a[i] ≤ 100
    - 1 ≤ b[i] ≤ 100

#### 문제풀이
- a,b의 배열의 원소의 크기를 확인후, 더 큰쪽에 점수를 +1씩 하는 함수를 작성
- 배열의 index를 for문으로 돌려서 if문으로 더 큰쪽에 +1를 하는 코드를 작성 후 해당 점수들을 return하는 함수를 작성하였다.


```python
def compareTriplets(a, b):
    alice = 0
    bob = 0
    for i in range(len(a)):
        if a[i] > b[i]:
            alice += 1
        elif a[i] < b[i]:
            bob += 1
    return alice, bob
```


```python
compareTriplets([1,2,3],[3,2,1])
```
    (1, 1)