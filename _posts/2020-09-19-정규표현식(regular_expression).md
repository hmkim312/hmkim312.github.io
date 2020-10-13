---
title:  정규표현식(Regular Express)
author: HyunMin Kim
date: 2020-09-19 16:30:00 0000
categories: [Python, Basic]
tags: [Regular Express]
---

## 1. Regular Express
---
### 1.1 연습용 문장

```python
import re
search_target = '''Luke Skywarker 02-123-4567 luke@daum.net
다스베이더 070-9999-9999 darth_vader@gmail.com 서울시 서초구 서초동 서초아파트
princess leia 010 2454 3457 leia@gmail.com'''
```

<br>

### 1.2 숫자가 연속해서 나타나는 첫 번째 문자열 찾기


```python
re.search('\d+', search_target).group()
```
    '02'

- search 메서드를 사용
- group : 매칭 문자열을 문자열로 반환
- \d+ : 숫자가 연속해서 나타남

<br>

### 1.3 000-000-0000의 형태를 찾아서 반환

```python
re.search('\d+-\d+-\d+', search_target).group()
```
    '02-123-4567'

- \d+-\d+-\d+ 을 search 메서드로 사용하여 찾음

<br>

### 1.4 전체 문자열에서 매칭되는 문자열 전부를 리스트형으로 반환

```python
re.findall('\d+-\d+-\d+', search_target)
```
    ['02-123-4567', '070-9999-9999']

- findall메서드를 사용하면 해당 조건에 맞는 단어들을 모두 찾아줌

<br>

### 1.5 메일 주소 형식 찾기

```python
re.findall('\w+@\w+[.]\w+', search_target)
```
    ['luke@daum.net', 'darth_vader@gmail.com', 'leia@gmail.com']

- \w+@ : @ 앞에 영문이 온다는 뜻
- \w+[.] : 영문 뒤에 .이 온다는 뜻
- \w+ : 영문
- 종합 : 영문@영문.영문

<br>

### 1.6 한글로 된 단어만 찾기


```python
re.findall('[가-힣]+',search_target)
```
    ['다스베이더', '서울시', '서초구', '서초동', '서초아파트']

- 한글은 [가-힣]으로 찾을수 있음

### 1.7 ```***시 ***구 ***동```으로 된 구성을 검색

```python
re.findall('[가-힣]+시\s[가-힣]+구\s[가-힣]+동',search_target)
```
    ['서울시 서초구 서초동']



- [가-힣]+시 : [한글]시를 뜻함
- \s[가-힣]+구 : 공백[한글]구 를 뜻함
- \s[가-힣]+동 : 공백[한글]동 을 뜻함

<br>

### 1.8 아파트로 된 단어의 위치 찾기

```python
re.search('[가-힣]+아파트',search_target).span()
```
    (95, 100)



- [가-힣] : `~~`아파트 된 단어
- span : 위치를 출력해줌
- search_target의 95 ~100번쨰에 해당 단어가 있다는 뜻

<br>

```python
search_target[95:100]
```
    '서초아파트'

- 해당 위치를 확인해보니 서초아파트가 나옴
