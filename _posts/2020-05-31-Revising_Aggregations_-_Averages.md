---
title: Revising Aggregations - Averages
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Where. Avg]
---

# Revising Aggregations - Averages

- URL : <https://www.hackerrank.com/challenges/revising-aggregations-the-average-function/problem>{:target="_blank"}
- Query the average population of all cities in CITY where District is California.
- MySQL Solution

```sql
select avg(population)
from city
where district = 'California'
```