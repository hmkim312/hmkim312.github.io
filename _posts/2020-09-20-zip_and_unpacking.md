---
title: Zip과 언팩킹
author: HyunMin Kim
date: 2020-09-20 23:30:00 0000
categories: [Datascience, Python]
tags: [Zip]
---

## 1. Zip과 언패킹
### 1.1 리스트를 튜플로 zip
- zip을 이용하여 두개의 list를 tuple형태로 묶을수 있다.
```python
list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]
```

```python
pairs = [pair for pair in zip(list1, list2)]
pairs
```
    [('a', 1), ('b', 2), ('c', 3)]



### 1.2 튜플을 dict으로
- 위에서 만든 tuple을 dict 명령어를 사용하여 dict으로 변환 가능
```python
dict(pairs)
```
    {'a': 1, 'b': 2, 'c': 3}

### 1.3 한번에 할수 있음

```python
dict(zip(list1, list2))
```
    {'a': 1, 'b': 2, 'c': 3}



### 1.4 언패킹 인자를 이용한 역변환
- zip(*)을 사용하여 역변환 할수 있다.

```python
a, b = zip(*pairs)
```


```python
print(list(a))
print(list(b))
```

    ['a', 'b', 'c']
    [1, 2, 3]



```python

```