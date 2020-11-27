---
title: String Validators (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/string-validators/problem>{:target="_blank"}

- Python has built-in string validation methods for basic data. It can check if a string is composed of alphabetical characters, alphanumeric characters, digits, etc.
- str.isalnum()
    - This method checks if all the characters of a string are alphanumeric (a-z, A-Z and 0-9).

```python
>>> print 'ab123'.isalnum()
True
>>> print 'ab123#'.isalnum()
False
```

- str.isalpha()
    - This method checks if all the characters of a string are alphabetical (a-z and A-Z).
```python
>>> print 'abcD'.isalpha()
True
>>> print 'abcd1'.isalpha()
False
```

- str.isdigit()
    - This method checks if all the characters of a string are digits (0-9).
```python
>>> print '1234'.isdigit()
True
>>> print '123edsd'.isdigit()
False
```

- str.islower()
    - This method checks if all the characters of a string are lowercase characters (a-z).
```python
>>> print 'abcd123#'.islower()
True
>>> print 'Abcd123#'.islower()
False
```

- str.isupper()
    - This method checks if all the characters of a string are uppercase characters (A-Z).
```python
>>> print 'ABCD123#'.isupper()
True
>>> print 'Abcd123#'.isupper()
False
```
- Task
    - You are given a string S.
    - Your task is to find out if the string S contains: alphanumeric characters, alphabetical characters, digits, lowercase and uppercase characters.

- Input Format
    - A single line containing a string S.

- Constraints
    - 0 < len(S) <= 1000

- Output Format
    - In the first line, print True if S has any alphanumeric characters. Otherwise, print False.
    - In the second line, print True if S has any alphabetical characters. Otherwise, print False.
    - In the third line, print True if S has any digits. Otherwise, print False.
    - In the fourth line, print True if S has any lowercase characters. Otherwise, print False.
    - In the fifth line, print True if S has any uppercase characters. Otherwise, print False.

#### 문제풀이
- 주어진 string S에서 isalnum, isalpha, isdigit, islower, isupper가 참이되는 문자가 있으면 true 반환 없으면 False
- any(x)는 반복 가능한(iterable) 자료형 x를 입력 인수로 받으며 이 x의 요소 중 하나라도 참이 있으면 True를 돌려주고, x가 모두 거짓일 때에만 False를 돌려준다. all(x)의 반대이다.
- 그래서 any 함수를 이용하여 for문의 리스트컴프리헨션을 사용하고 True가 하나라도 있으면 any함수를 통해 True가 반환되게 짬
- 솔직히 any함수가 있었는지 몰랐는데, 이번기회에 알게되었음


```python
if __name__ == '__main__':
    s = input()
    print(any([c.isalnum() for c in s]))
    print(any([c.isalpha() for c in s]))
    print(any([c.isdigit() for c in s]))
    print(any([c.islower() for c in s]))
    print(any([c.isupper() for c in s]))
```

     qA2


    True
    True
    True
    True
    True
