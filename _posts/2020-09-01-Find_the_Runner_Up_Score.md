---
title: Find the Runner-Up (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/find-second-maximum-number-in-a-list/problem>{:target="_blank"}

- Given the participants' score sheet for your University Sports Day, you are required to find the runner-up score. You are given  scores. Store them in a list and find the score of the runner-up.

- Input Format
    - The first line contains n. The second line contains an array A[] of n integers each separated by a space.

- Constraints
    - 2 <= n <= 10
    - -100 <= A[i] <= 100

- Output Format
    - Print the runner-up score.

#### 문제 풀이
- 주어진 배열에서 준우승을 출력하는것
- 배열을 set으로 중복제거 후 sort로 정렬하여 맨뒤에 하나를(최고점수) pop으로 제거 후 마지막것을 출력  


```python
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    arr = list(arr)
    ls= list(set(arr))
    ls.sort()
    ls.pop()
    print(ls[-1])
```




    5
