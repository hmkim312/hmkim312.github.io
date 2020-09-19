---
title: Regular Express
author: HyunMin Kim
date: 2020-09-19 16:30:00 0000
categories: [Datascience, Python]
tags: [Regular Express, Study]
---



## 1. Regular Express

### 1.1 연습용 문장


```python
import re
search_target = '''Luke Skywarker 02-123-4567 luke@daum.net
다스베이더 070-9999-9999 darth_vader@gmail.com 서울시 서초구 서초동 서초아파트
princess leia 010 2454 3457 leia@gmail.com'''
```

### 1.2 search_target에서 숫자가 연속해서 나타나는 첫 번째 매칭 문자열을 문자열로 반환


```python
re.search('\d+', search_target).group()
```




    '02'



### 1.3 search_target에서 연속된 숫자(\d+)가 – 기호를 두 번 사이에 끼고 세 번 반복하는 즉, 000-000-0000의 형태를 찾아서 반환


```python
re.search('\d+-\d+-\d+', search_target).group()
```




    '02-123-4567'



### 1.4 전체 문자열에서 매칭되는 문자열 전부를 리스트형으로 반환


```python
re.findall('\d+-\d+-\d+', search_target)
```




    ['02-123-4567', '070-9999-9999']



### 1.5 메일 주소 형식을 찾아서 전체 매칭을 반환


```python
re.findall('\w+@\w+[.]\w+', search_target)
```




    ['luke@daum.net', 'darth_vader@gmail.com', 'leia@gmail.com']



### 1.6 한글로 된 단어만 찾기


```python
re.findall('[가-힣]+',search_target)
```




    ['다스베이더', '서울시', '서초구', '서초동', '서초아파트']



### 1.7 ***시 ***구 ***동으로 된 구성을 검색


```python
re.findall('[가-힣]+시\s[가-힣]+구\s[가-힣]+동',search_target)
```




    ['서울시 서초구 서초동']



### 1.8 아파트로 된 단어의 위치 span()을 찾음


```python
re.search('[가-힣]+아파트',search_target).span()
```




    (95, 100)




```python
search_target[95:100]
```




    '서초아파트'




```python

```
