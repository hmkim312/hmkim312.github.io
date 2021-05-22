---
title: n부터 m까지 자연수의 합 [CosProLv2]
author: HyunMin Kim
date: 2021-05-22 00:00:00 0000
categories: [Programers, CosProLv2 Part1]
tags: [Programers, CosProLv2 Part1]
---

URL <https://programmers.co.kr/learn/courses/33/lessons/1855>{:target="_blank"}

### 문제 설명
- 두 자연수 n부터 m까지의 합을 구하려고 합니다. 이를 위해 다음과 같이 3단계로 간단히 프로그램 구조를 작성했습니다.
1. 1부터 m까지의 합을 구합니다.
2. 1부터 n-1까지의 합을 구합니다.
3. 1번 단계에서 구한 값에서 2번 단계에서 구한 값을 뺍니다.
- 두 자연수 n과 m이 매개변수로 주어질 때, n 부터 m 까지의 합을 return 하도록 solution 함수를 작성했습니다. 이때, 위 구조를 참고하여 중복되는 부분은 func_a라는 함수로 작성했습니다. 코드가 올바르게 동작할 수 있도록 빈칸을 알맞게 채워주세요.

#### 매개변수 설명
- 두 자연수 n과 m이 solution 함수의 매개변수로 주어집니다.
- n, m은 1 이상 10,000 이하의 자연수이며, 항상 n ≤ m 을 만족합니다.

#### return 값 설명
- solution 함수는 n부터 m까지의 합을 return 합니다.

#### 빈칸 코드

```python
def func_a(k):
    sum = 0
    for i in range('빈칸'):
        sum += '빈칸'

    return sum

def solution(n, m):
    sum_to_m = func_a(m)
    sum_to_n = func_a(n-1)
    answer = sum_to_m - sum_to_n
    return answer
```

#### 문제 풀이
- 빈칸이 있는 코드에 알맞은 빈칸을 넣으면 된다.
- 일단 주어진 수에서 0 ~ 주어진수까지의 모두 더하는 func_a에서 빈칸이 있는대, range는 k보다 1작은수 까지만 만들어짐, 그래서 주어진 k에서 +1를 함
- i를 모두 더해주면 total sum을 하게 됨


```python
def func_a(k):
    sum = 0
    for i in range(k+1):
        sum += i
    return sum

def solution(n, m):
    sum_to_m = func_a(m)
    sum_to_n = func_a(n-1)
    answer = sum_to_m - sum_to_n
    return answer
```


```python
n=5
m=10
solution(n, m)
```




    45




```python
n=6
m=6
solution(n, m)
```




    6


