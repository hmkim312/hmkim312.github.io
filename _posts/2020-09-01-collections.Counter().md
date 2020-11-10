---
title: Collections.Counter() (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/collections-counter/problem>{:target="_blank"}

- A counter is a container that stores elements as dictionary keys, and their counts are stored as dictionary values.
- Sample Code
```python
>>> from collections import Counter
>>> 
>>> myList = [1,1,2,3,4,5,3,2,3,4,2,1,2,3]
>>> print Counter(myList)
Counter({2: 4, 3: 4, 1: 3, 4: 2, 5: 1})
>>>
>>> print Counter(myList).items()
[(1, 3), (2, 4), (3, 4), (4, 2), (5, 1)]
>>> 
>>> print Counter(myList).keys()
[1, 2, 3, 4, 5]
>>> 
>>> print Counter(myList).values()
[3, 4, 4, 2, 1]
```

- Task
    - Ragha is a shoe shop owner. His shop has X number of shoes.
    - He has a list containing the size of each shoe he has in his shop.
    - There are N number of customers who are willing to pay x; amount of money only if they get the shoe of their desired size.
    - Your task is to compute how much money Ragha earned.

- Input Format
    - The first line contains X, the number of shoes.
    - The second line contains the space separated list of all the shoe sizes in the shop.
    - The third line contains N, the number of customers.
    - The next N lines contain the space separated values of the  desired by the customer and , the price of the shoe.

- Constraints
    - 0 < X < 10^3
    - 0 < N <= 10^3
    - 20 < xi < 100
    - 2 < shoe size < 20

- Output Format
    - Print the amount of money earned by Ragha.

#### 문제풀이
- 신발의 사이즈가 적힌 리스트를 받고, 해당 사이즈를 구매하는 고객들의 가격의 총합을 구하는것
- 사이즈가 적힌 리스트는 input으로 받으면서 Counter 함수로 dict 형태의 s를 만들어준다
- 고객의 수만큼 for문을 돌리면서 s 안에 size가 있다면 해당 size의 price를 더해주고, 해당 size의 갯수는 1만큼 줄여준다
- 이를 반복 한뒤 마지막 최종 price를 출력하면 끝


```python
from collections import Counter
x = int(input())
s = Counter(map(int,input().split()))
n = int(input())
total = 0
for i in range(n):
    size, price = map(int, input().split())
    if s[size]:
        total += price
        s[size] -= 1
print(total)
```

     10
     2 3 4 5 6 8 7 6 5 18
     6
     6 55
     6 45
     6 55
     4 40
     18 60
     10 50


    200
