---
title: Population Density Difference
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Oracle]
---

# Population Density Difference

- URL : <https://www.hackerrank.com/challenges/population-density-difference/problem>{:target="_blank"}
- Query the difference between the maximum and minimum populations in CITY.
- MySQL, Oracle Solution

```sql
select max(population) - min(population)
from city;
```