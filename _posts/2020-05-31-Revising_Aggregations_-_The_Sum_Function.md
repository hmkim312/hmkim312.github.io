---
title: Revising Aggregations - The Sum Function
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Where, Sum]
---


# Revising Aggregations - The Sum Function

- URL : <https://www.hackerrank.com/challenges/revising-aggregations-sum/problem>{:target="_blank"}
- Query the total population of all cities in CITY where District is California.
- MySQL Solution

```sql
select sum(population)
from city
where district = 'California'
```