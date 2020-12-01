---
title: Text_Alignment (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/text-alignment/problem>{:target="_blank"}

- In Python, a string of text can be aligned left, right and center.
- .ljust(width)
    - This method returns a left aligned string of length width.
    
```python
>>> width = 20
>>> print 'HackerRank'.ljust(width,'-')
HackerRank----------  
```
- .center(width)
    - This method returns a centered string of length width.
    
```python
>>> width = 20
>>> print 'HackerRank'.center(width,'-')
-----HackerRank-----
```

- .rjust(width)
    - This method returns a right aligned string of length width.
    
```python
>>> width = 20
>>> print 'HackerRank'.rjust(width,'-')
----------HackerRank
```
- Task
    - You are given a partial code that is used for generating the HackerRank Logo of variable thickness.
    - Your task is to replace the blank (______) with rjust, ljust or center.

- Input Format
    - A single line containing the thickness value for the logo.

- Constraints
    - The thickness must be an odd number.
    - 0 < thickness <= 50

- Output Format
    - Output the desired logo.

#### 문제풀이
- .ljust(width), .center(width) .rjust(width)를 이용하여 ____를 알맞은 함수로 변경하는 문제
- 처음엔 무슨소리가 싶긴했는데, 해당 문제의 output을 잘보며 center, ljust, rjust를 잘 이용하여 풀면 쉬웠다.


```python
#Replace all ______ with rjust, ljust or center. 

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).______(thickness-1)+c+(c*i).______(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).______(thickness*2)+(c*thickness).______(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).______(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).______(thickness*2)+(c*thickness).______(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).______(thickness)+c+(c*(thickness-i-1)).______(thickness)).______(thickness*6))
```


```python
#Replace all ______ with rjust, ljust or center. oK

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))
```

     5


        H    
       HHH   
      HHHHH  
     HHHHHHH 
    HHHHHHHHH
      HHHHH               HHHHH             
      HHHHH               HHHHH             
      HHHHH               HHHHH             
      HHHHH               HHHHH             
      HHHHH               HHHHH             
      HHHHH               HHHHH             
      HHHHHHHHHHHHHHHHHHHHHHHHH   
      HHHHHHHHHHHHHHHHHHHHHHHHH   
      HHHHHHHHHHHHHHHHHHHHHHHHH   
      HHHHH               HHHHH             
      HHHHH               HHHHH             
      HHHHH               HHHHH             
      HHHHH               HHHHH             
      HHHHH               HHHHH             
      HHHHH               HHHHH             
                        HHHHHHHHH 
                         HHHHHHH  
                          HHHHH   
                           HHH    
                            H     



```python

```
