---
title: Draw The Triangle 2
author: HyunMin Kim
date: 2020-11-02 15:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL]
---

# Draw The Triangle 2

- URL : <https://www.hackerrank.com/challenges/draw-the-triangle-2/problem>{:target="_blank"}
- P(R) represents a pattern drawn by Julia in R rows. The following pattern represents P(5):

```
* 
* * 
* * * 
* * * * 
* * * * *
```

- Write a query to print the pattern P(20).
- MySQL Solution

```sql
set @number = 0;
select repeat('* ', @number := @number + 1) 
from information_schema.tables
where @number < 20
```