---
title: Day08 - Dictionaries and Maps (Python 3)
author: HyunMin Kim
date: 2020-06-05 00:00:00 0000
categories: [Hacker Ranker, 30 Days of Code]
tags: [While]
---

- URL : <https://www.hackerrank.com/challenges/30-dictionaries-and-maps/problem>{:target="_blank"}

- Objective
    - Today, we're learning about Key-Value pair mappings using a Map or Dictionary data structure. Check out the Tutorial tab for learning materials and an instructional video!

- Task
    - Given n names and phone numbers, assemble a phone book that maps friends' names to their respective phone numbers. You will then be given an unknown number of names to query your phone book for. For each  name queried, print the associated entry from your phone book on a new line in the form name=phoneNumber; if an entry for name is not found, print Not found instead.

- Note: Your phone book should be a Dictionary/Map/HashMap data structure.

- Input Format

    - The first line contains an integer, n, denoting the number of entries in the phone book.
    - Each of the n subsequent lines describes an entry in the form of 2 space-separated values on a single line. The first value is a friend's name, and the second value is an 8-digit phone number.
    - After the n lines of phone book entries, there are an unknown number of lines of queries. Each line (query) contains a name to look up, and you must continue reading lines until there is no more input.

- Note: Names consist of lowercase English alphabetic letters and are first names only.

- Constraints
    - 1 <= n <= 10^5
    - 1 <= queries <= 10^5
- Output Format
    - On a new line for each query, print Not found if the name has no corresponding entry in the phone book; otherwise, print the full name and phonenumber in the format name=phoneNumber.


```python
# for문 대신 while문으로 런타임 오류 해결
n = int(input())
phonebooks_dict = {}
for i in range(n):
    phonebook = input().split()
    phonebooks_dict[phonebook[0]] = phonebook[1]

while True:
    try:
        k = input()
        result = k+'='+phonebooks_dict[k] if k in phonebooks_dict else 'Not found'
        print(result)
    except EOFError:
        break
```


```python
# for문을 2번 돌리니, case1에서 runtime error가 뜸
n = int(input())
phonebooks_dict = {}
for i in range(n):
    phonebook = input().split()
    phonebooks_dict[phonebook[0]] = phonebook[1]

for j in range(n):
    k = input()
    result = k+'='+phonebooks_dict[k] if k in phonebooks_dict else 'Not found'
    print(result)
```
