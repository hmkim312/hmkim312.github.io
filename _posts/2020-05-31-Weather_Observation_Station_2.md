---
title: Weather Observation Station 2
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Oracle, Where]
---

# Weather Observation Station 2

- URL : <https://www.hackerrank.com/challenges/weather-observation-station-2/problem>{:target="_blank"}
- Query the following two values from the STATION table:
- The sum of all values in LAT_N rounded to a scale of  decimal places.
- The sum of all values in LONG_W rounded to a scale of  decimal places.
- MySQL, Oracle Solution

```sql
select name
from city
where Population > 120000 and countrycode = 'USA';
```