---
title: Revising Aggregations - The Count Function
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Oracle, Where, Count]
---


# Revising Aggregations - The Count Function

- URL : <https://www.hackerrank.com/challenges/revising-aggregations-the-count-function/problem>{:target="_blank"}
- Query a count of the number of cities in CITY having a Population larger than 100,000.
- MySQL, Oracle Solution

```sql
select count(name)
from city
where population >= 100000;
```