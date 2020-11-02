---
title: Employee Names
author: HyunMin Kim
date: 2020-05-31 16:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Order By]
---

# Employee Names

- URL : <https://www.hackerrank.com/challenges/name-of-employees/problem>{:target"_blank"}
- Write a query that prints a list of employee names (i.e.: the name attribute) from the Employee table in alphabetical order.
- MySQL Solution

```sql
select name
from Employee
order by name asc;
```