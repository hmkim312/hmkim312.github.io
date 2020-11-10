---
title: Find a string (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: [Startswith]
---

- URL : <https://www.hackerrank.com/challenges/find-a-string/problem>{:target="_blank"}

- In this challenge, the user enters a string and a substring. You have to print the number of times that the substring occurs in the given string. String traversal will take place from left to right, not from right to left.

- NOTE: String letters are case-sensitive.

- Input Format
    - The first line of input contains the original string. The next line contains the substring.

- Constraints
    - 1 <= len(stirng) <= 200
    - Each character in the string is an ascii character.

- Output Format
    - Output the integer number indicating the total number of occurrences of the substring in the original string.

#### 문제풀이
- 첫번째 문자가 주어지고, 두번째 문자가 주어졌을때 첫번쨰 문자가 두번째 문자로 시작하는 횟수를 반환
- python의 startswith는 문자열이 특정문자로 시작하는지 여부를 알려줌, 해당 함수를 이용하여 for문으로 코드 작성


```python
a = 'ABCDCDC'
```


```python
b = 'CDC'
```


```python
# startswith는 문자열이 특정문자로 시작하는지 여부를 알려준다

def count_substring(string, sub_string):
    count = 0
    for i in range(0, len(string)):
        if string[i:].startswith(sub_string):
            count += 1
    return count

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)
```

    2

