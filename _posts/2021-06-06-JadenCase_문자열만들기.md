---
title: JadenCase [Python]
author: HyunMin Kim
date: 2021-06-06 00:00:00 0000
categories: [Programers, Python Lv2]
tags: [Programers, Python Lv2]
---

URL <https://programmers.co.kr/learn/courses/30/lessons/12951>:{target="_blank"}

#### 문제 설명
- JadenCase란 모든 단어의 첫 문자가 대문자이고, 그 외의 알파벳은 소문자인 문자열입니다. 문자열 s가 주어졌을 때, s를 JadenCase로 바꾼 문자열을 리턴하는 함수, solution을 완성해주세요.

#### 제한 조건
- s는 길이 1 이상인 문자열입니다.
- s는 알파벳과 공백문자(" ")로 이루어져 있습니다.
- 첫 문자가 영문이 아닐때에는 이어지는 영문은 소문자로 씁니다. ( 첫번째 입출력 예 참고 )

#### 문제 풀이
- python의 str 메서드 중에 대문자를 만드는 함수를 사용하면됨
- title은 첫문자가 영문이 아닐떄는 소문자로 쓰는 이유떄문에 capitalize를 사용함
- 주어진 s를 공백으로 split함 (2번째 조건 공백문자도 있기 때문)
- 이후 주어진 s를 for문을 돌며 capitalize로 맨 앞에만 대문자로 만듬
- capitalize는 첫 문자가 영문이 아니면 대문자가 되지 않음
- 이를 temp 리스트에 넣고 마지막에 공백으로 join한 문자열을 return해주면됨

#### 추가
- 그냥 string 메서드의 capwords를 sep = ' '으로 사용하면됨.
- capwords가 captitalize + split + join을 한번에 해주는 메서드


```python
def solution(s):
    s_s = s.split(' ')
    temp = []
    for i in s_s:
        i = i.capitalize()
        temp.append(i)
    return ' '.join(temp)
```


```python
s = '3people unFollowed me'
solution(s)
```




    '3people Unfollowed Me'




```python
s = ' 3people unFollowed me '
solution(s)
```




    ' 3people Unfollowed Me '



#### capwords를 사용함


```python
import string
def solution_2(s):
    return string.capwords(s, sep = ' ')
```


```python
s = ' 3people unFollowed me '
solution_2(s)
```




    ' 3people Unfollowed Me '




```python
s = '3people unFollowed me'
solution_2(s)
```




    '3people Unfollowed Me'


