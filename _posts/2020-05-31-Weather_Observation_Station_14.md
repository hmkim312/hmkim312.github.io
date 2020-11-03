---
title: Weather Observation Station 14
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Oracle, Round, Where, Max]
---

# Weather Observation Station 14

- URL : <https://www.hackerrank.com/challenges/weather-observation-station-14/problem>{:target="_blank"}
- Query the greatest value of the Northern Latitudes (LAT_N) from STATION that is less than 137.2345. Truncate your answer to 4 decimal places.
- MySQL, Oracle Solution

```sql
select round(max(lat_n), 4)
from station
where lat_n < 137.2345;
```