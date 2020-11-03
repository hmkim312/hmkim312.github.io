---
title: Type of Triangle
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Oracle, Case, When]
---

# Type of Triangle

- URL : <https://www.hackerrank.com/challenges/what-type-of-triangle/problem>{:target="_blank"}
- Write a query identifying the type of each record in the TRIANGLES table using its three side lengths. Output one of the following statements for each record in the table:
- Equilateral: It's a triangle with 3 sides of equal length.
- Isosceles: It's a triangle with 2 sides of equal length.
- Scalene: It's a triangle with 3 sides of differing lengths.
- Not A Triangle: The given values of A, B, and C don't form a triangle.
- MySQL, Oracle Solution

```sql
select 
    case 
        when a + b <= c or a + c <= b or b + c <= A then 'Not A Triangle'
        when a = b and a = c then 'Equilateral'
        when a = b or a = c or b = c then 'Isosceles'
        else 'Scalene'
    end
from triangles;
```