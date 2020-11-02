---
title: Draw The Triangle 1
author: HyunMin Kim
date: 2020-05-31 14:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL]
---

# Draw The Triangle 1

- URL : <https://www.hackerrank.com/challenges/draw-the-triangle-1/problem>{:target="_blank"}
- P(R) represents a pattern drawn by Julia in R rows. The following pattern represents P(5):

```
* * * * * 
* * * * 
* * * 
* * 
*
```

- Write a query to print the pattern P(20).
- MySQL Solution

```sql
set @number = 21;
select repeat('* ', @number := @number - 1) 
from information_schema.tables;
```