---
title: Weather Observation Station 16
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Round, Where, Min]
---

# Weather Observation Station 16

- URL : <https://www.hackerrank.com/challenges/weather-observation-station-16/problem>{:target="_blank"}
- Query the smallest Northern Latitude (LAT_N) from STATION that is greater than 38.7780. Round your answer to 4 decimal places.
- MySQL Solution

```sql
select round(min(lat_n),4)
from station
where lat_n > 38.7780
```