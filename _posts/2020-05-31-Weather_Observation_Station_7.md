---
title: Weather Observation Station 7
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Like, Where]
---

# Weather Observation Station 7

- URL : <https://www.hackerrank.com/challenges/weather-observation-station-7/problem>{:target="_blank"}
- Query the list of CITY names ending with vowels (a, e, i, o, u) from STATION. Your result cannot contain duplicates.
- MySQL Solution

```sql
select distinct(city)
from station
where (city like '%A' or
       city like '%E' or
       city like '%I' or
       city like '%O' or
       city like '%U' )
```