---
title: Average Population
author: HyunMin Kim
date: 2020-05-31 13:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Oracle]
---

# Average Population

- URL : <https://www.hackerrank.com/challenges/average-population/problem>{:target="_blank"}
- Query the average population for all cities in CITY, rounded down to the nearest integer.
- MySQL, Oracle Solutaion

```sql
select round(avg(population))
from city
```