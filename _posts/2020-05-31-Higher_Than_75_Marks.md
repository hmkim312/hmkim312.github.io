---
title: Higher Than 75 Marks
author: HyunMin Kim
date: 2020-05-31 18:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Order By, Where]
---

# Higher Than 75 Marks

- URL : <https://www.hackerrank.com/challenges/more-than-75-marks/problem>{:target="_blank"}
- Query the Name of any student in STUDENTS who scored higher than 75 Marks. Order your output by the last three characters of each name. If two or more students both have names ending in the same last three characters (i.e.: Bobby, Robby, etc.), secondary sort them by ascending ID.
- MySQL Solution

```sql
select name
from students
where marks > 75
order by right(name, 3), id asc;
```