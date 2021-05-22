---
title: 이름에 j또는 k가 들어가는 사람 [CosProLv2]
author: HyunMin Kim
date: 2021-05-22 00:00:00 0000
categories: [Programers, CosProLv2 Part2]
tags: [Programers, CosProLv2 Part2]
---

URL <https://programmers.co.kr/learn/courses/33/lessons/1861>{:target="_blank"}

#### 문제 설명
- 학생들의 이름이 들어있는 명단에서 이름에 j 또는 k가 들어가는 학생의 수를 구하려고 합니다.
- 예를 들어 "james"에는 j가 들어가 있으며, "jack"에는 j와 k가 모두 들어있습니다.
- 학생들의 이름이 들어있는 배열 name_list가 매개변수로 주어졌을 때, 이름에 j 또는 k가 들어가는 학생의 수를 세서 return 하도록 solution 함수를 작성했습니다. 그러나, 코드 일부분이 잘못되어있기 때문에, 몇몇 입력에 대해서는 올바르게 동작하지 않습니다. 주어진 코드에서 한 줄만 변경해서 모든 입력에 대해 올바르게 동작하도록 수정하세요.

#### 매개변수 설명
- 학생들의 이름이 들어있는 배열 name_list가 solution 함수의 매개변수로 주어집니다.
- name_list의 길이는 1 이상 100 이하입니다.
- 학생들의 이름은 알파벳 소문자로만 이루어져 있으며, 길이는 1 이상 20 이하입니다.
- 같은 이름이 중복해서 들어있지 않습니다.

#### return 값 설명
- solution 함수는 이름에 j 또는 k가 들어가는 학생의 수를 return 합니다.

#### 수정할 코드
```python
def solution(name_list):
    answer = 0
    for name in name_list:
        for n in name:
            if n == 'j' or n == 'k':
                answer += 1
                continue
    return answer
```

#### 문제 풀이
- 주어진 코드에서 1줄만 수정하여 정상작동하는 코드를 만들어내는 문제이다.
- 2번째 for문에서 name이 n으로 나올때 continue를 사용하게 되면 이름을 계속 보게되는것이라 Jack같은 경우는 answer가 +2가 되버림
- 그래서 2번째 for문에서 name의 원소들 n 중에 j or k를 만나게 되면 break를 걸어 stop하게 하고 첫번째 for문으로 돌아가게 함


```python
def solution(name_list):
    answer = 0
    for name in name_list:
        for n in name:
            if n == 'j' or n == 'k':
                answer += 1
                break
    return answer
```


```python
name_list = ["james", "luke", "oliver", "jack"]
solution(name_list)
```




    3


