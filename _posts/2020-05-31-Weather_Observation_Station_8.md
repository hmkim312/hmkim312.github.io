---
title: Weather Observation Station 8
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Distinct, Where]
---

# Weather Observation Station 8

- URL : <https://www.hackerrank.com/challenges/weather-observation-station-8/problem>{:target="_blank"}
- Query the list of CITY names from STATION which have vowels (i.e., a, e, i, o, and u) as both their first and last characters. Your result cannot contain duplicates.
- MySQL Solution

```sql
select distinct(city)
from station
where left(city, 1) in ('a','e','i','o','u') and
      right(city, 1) in ('a','e','i','o','u')
```