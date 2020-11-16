---
title: Nested Lists (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/nested-list/problem>{:target="_blank"}

- Given the names and grades for each student in a class of N students, store them in a nested list and print the name(s) of any student(s) having the second lowest grade.

- Note: If there are multiple students with the second lowest grade, order their names alphabetically and print each name on a new line.

- Example
    - recode = [['chi',20.0],['beta','50.0],['alpha', 50.0]]

    - The ordered list of scores is [20.0, 50.0], so the second lowest score is 50.0. There are two students with that score: ['beta','alpha']. Ordered alphabetically, the names are printed as:
    
```python
alpha
beta
```

- Input Format

- The first line contains an integer, N, the number of students.
- The 2N subsequent lines describe each student over 2 lines.
    - The first line contains a student's name.
    - The second line contains their grade.

- Constraints
    - 2 <= N <= 5
    - There will always be one or more students having the second lowest grade.
    
- Output Format
    - Print the name(s) of any student(s) having the second lowest grade in. If there are multiple students, order their names alphabetically and print each one on a new line.

#### 문제풀이
- 학생과 점수가 주어졌을때 2번째로 낮은 점수의 학생 이름을 출력하기 (동점자가 있을경우 이름순으로 출력)
- 전체 recodes를 담을 리스트를 생성 후, 입력받는 name과 score를 저장
- 전체 recodes를 학생 이름순으로 정렬 (동점자가 있을 경우 이름순으로 출력하기 위해)
- 전체 recodes에서 아래에서 2등 점수를 가져옴 (set을 하는 이유는 꼴등이 여러명일 경우가 있을수 있으므로)
    - [score for name, score in recodes] : recodes에서 이름과 점수로 분리 후 점수만 list에 저장
    - set[score for name, score in recodes] : list에 저장한것을 다시 set(중복제거)
    - sorted(list(set([score for name, score in recodes])))[1] : 중복제거된 집합을 다시 list로 바꾸고, 정렬시킨뒤 그중 2번째애만 저장
    - 위의 점수가 score가 같으면 name을 출력하게 함


```python
recodes = []
if __name__ == '__main__':
    for _ in range(int(input())):
        name = input()
        score = float(input())
        recodes.append([name, score])
    recodes = sorted(recodes)
    for name, socre in recodes:
        if socre == sorted(list(set([score for name, score in recodes])))[1]:
            print(name)
```

     5
     Harry
     37.21
     Berry
     37.21
     Tina
     37.2
     Akriti
     41
     Harsh
     39


    Berry
    Harry



```python

```
