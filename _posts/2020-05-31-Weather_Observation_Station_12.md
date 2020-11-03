---
title: Weather Observation Station 12
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Distinct, Where]
---

# Weather Observation Station 12

- URL : <https://www.hackerrank.com/challenges/weather-observation-station-12/problem>{:target="_blank"}
- Query the list of CITY names from STATION that do not start with vowels and do not end with vowels. Your result cannot contain duplicates.
- MySQL Solution

```sql
select distinct(city)
from station
where right(city, 1) not in ('a','e','i','o','u') and
      left(city, 1) not in ('a','e','i','o','u')
```