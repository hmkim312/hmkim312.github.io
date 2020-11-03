---
title: Weather Observation Station 5
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Length, Limit]
---

# Weather Observation Station 5

- URL : <https://www.hackerrank.com/challenges/weather-observation-station-5/problem>{:target="_blank"}
- Query the two cities in STATION with the shortest and longest CITY names, as well as their respective lengths (i.e.: number of characters in the name). If there is more than one smallest or largest city, choose the one that comes first when ordered alphabetically.
- MySQL Solution


```sql
select city, length(city) 
from station 
order by length(city) asc, city asc limit 1; 

select city, length(city) 
from station 
order by length(city) desc, city asc limit 1;
```