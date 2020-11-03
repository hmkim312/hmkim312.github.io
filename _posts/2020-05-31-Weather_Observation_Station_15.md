---
title: Weather Observation Station 15
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Round, Where, Max, Sub Query]
---

# Weather Observation Station 15

- URL : <https://www.hackerrank.com/challenges/weather-observation-station-15/problem>{:target="_blank"}
- Query the Western Longitude (LONG_W) for the largest Northern Latitude (LAT_N) in STATION that is less than 137.2345. Round your answer to 4 decimal places.
- MySQL Solution

```sql
select round(long_w, 4)
from station
where lat_n = (select max(lat_n) 
     from station
     where lat_n < 137.2345)
```