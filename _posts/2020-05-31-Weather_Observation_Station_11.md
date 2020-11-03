---
title: Weather Observation Station 11
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Distinct, Where]
---

# Weather Observation Station 11

- URL : <https://www.hackerrank.com/challenges/weather-observation-station-11/problem>{:target="_blank"}
- Query the list of CITY names from STATION that either do not start with vowels or do not end with vowels. Your result cannot contain duplicates.
- MySQL Solution

```sql
select distinct(city)
from station
where right(city, 1) not in ('a','e','i','o','u') or
      left(city, 1) not in ('a','e','i','o','u')
```