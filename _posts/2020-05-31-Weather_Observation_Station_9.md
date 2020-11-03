---
title: Weather Observation Station 9
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Distinct, Where]
---

# Weather Observation Station 9

- URL : <https://www.hackerrank.com/challenges/weather-observation-station-9/problem>{:target="_blank"}
- Query the list of CITY names from STATION that do not start with vowels. Your result cannot contain duplicates.
- MySQl Solution

```sql
select distinct(city)
from station
where left(city, 1) not in ('a','e','i','o','u');
```