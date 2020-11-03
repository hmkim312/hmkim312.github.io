---
title: Weather Observation Station 3
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Distinct]
---

# Weather Observation Station 3

- URL : <https://www.hackerrank.com/challenges/weather-observation-station-3/problem>{:target="_blank"}
- Query a list of CITY names from STATION for cities that have an even ID number. Print the results in any order, but exclude duplicates from the answer.
- MySQL Solution

```sql
select distinct(city)
from station
where (id % 2) = 0;
```