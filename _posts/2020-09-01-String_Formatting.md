---
title: String Formatting (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/python-string-formatting/problem>{:target="_blank"}

- Given an integer, n, print the following values for each integer i from 1 to n:
    - 1. Decimal
    - 2. Octal
    - 3. Hexadecimal (capitalized)
    - 4. Binary

- The four values must be printed on a single line in the order specified above for each i from 1 to n. Each value should be space-padded to match the width of the binary value of n.

- Input Format
    - A single integer denoting .

- Constraints
    - 1 <= n <= 99
    
- Output Format
    - Print n lines where each line i (in the range 1 <= i <= n) contains the respective decimal, octal, capitalized hexadecimal, and binary values of . Each printed value must be formatted to the width of the binary value of n.

#### 문제풀이
- 주어진 n에 대하여 1 ~ n까지 10진법, 8진법, 2진법, 16진법으로 각각 순서대로 출력하여, 각 공객은 n의 17진수의 길이(len)만큼 띄움
- format함수를 사용하여 숫자들을 바꾸고, 공백은 format함수로 변환 후 rjust로 주었다.


```python
def print_formatted(number):
    w = len(format(number, 'b'))
    for i in range(1, number+1):
        print(str(i).rjust(w,' '), format(i, 'o').rjust(w,' '), format(i, 'x').rjust(w,' ').upper(), format(i, 'b').rjust(w,' '))
        
if __name__ == '__main__':
    n = int(input())
    print_formatted(n)
```

     17


        1     1     1     1
        2     2     2    10
        3     3     3    11
        4     4     4   100
        5     5     5   101
        6     6     6   110
        7     7     7   111
        8    10     8  1000
        9    11     9  1001
       10    12     A  1010
       11    13     B  1011
       12    14     C  1100
       13    15     D  1101
       14    16     E  1110
       15    17     F  1111
       16    20    10 10000
       17    21    11 10001



```python
print_formatted(17)
```

        1     1     1     1
        2     2     2    10
        3     3     3    11
        4     4     4   100
        5     5     5   101
        6     6     6   110
        7     7     7   111
        8    10     8  1000
        9    11     9  1001
       10    12     A  1010
       11    13     B  1011
       12    14     C  1100
       13    15     D  1101
       14    16     E  1110
       15    17     F  1111
       16    20    10 10000
       17    21    11 10001



```python
number = 17
```


```python
str(17).rjust(w,' ')
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-49-525c88a013e8> in <module>
    ----> 1 str(17).rjust(w,' ')
    

    TypeError: 'str' object cannot be interpreted as an integer



```python
format(17, 'b')
```




    '10001'




```python
print(format(17, 'b'),sep = len(format(17, 'b')) * ' ')
```

    10001



```python
w = len(str(bin(number)).replace('0b',''))
d = str(17).rjust(w,' ')
```


```python
type(bin(number))
```




    str




```python
format(number, 'b')
```




    '10001'




```python

```


```python

```
