---
title: Weather Observation Station 17
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Round, Where, Sub Query]
---

# Weather Observation Station 17

- URL : <https://www.hackerrank.com/challenges/weather-observation-station-17/problem>{:target="_blank"}
- Query the Western Longitude (LONG_W)where the smallest Northern Latitude (LAT_N) in STATION is greater than 38.7780. Round your answer to 4 decimal places.
- MySQL Solution

```sql
select round(long_w,4)
from station
where lat_n = (select min(lat_n)
              from station
              where lat_n > 38.7780)
```