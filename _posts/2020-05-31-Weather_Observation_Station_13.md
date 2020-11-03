---
title: Weather Observation Station 13
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Round, Where, Between A and B]
---

# Weather Observation Station 13

- URL : <https://www.hackerrank.com/challenges/weather-observation-station-13/problem>{:target="_blank"}
- Query the sum of Northern Latitudes (LAT_N) from STATION having values greater than 38.7880 and less than 137.2345. Truncate your answer to 4 decimal places.
- MySQL Solution

```sql
select round(sum(lat_n),4)
from station
where lat_n between 38.7880 and 137.2345
```