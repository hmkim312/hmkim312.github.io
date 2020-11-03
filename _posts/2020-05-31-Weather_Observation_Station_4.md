---
title: Weather Observation Station 4
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Count, Distinct]
---

# Weather Observation Station 4

- URL : <https://www.hackerrank.com/challenges/weather-observation-station-4/problem>{:target="_blank"}
- Find the difference between the total number of CITY entries in the table and the number of distinct CITY entries in the table.
- MySQL, Oracle Solution

```sql
select (count(city) - count(distinct city))
from station;
```