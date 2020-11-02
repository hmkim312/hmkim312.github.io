---
title: Average Population of Each Continent
author: HyunMin Kim
date: 2020-05-31 12:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Join, Group By]
---

# Average Population of Each Continent

- URL : <https://www.hackerrank.com/challenges/average-population-of-each-continent/problem>{:target="_blank"}
- Given the CITY and COUNTRY tables, query the names of all the continents (COUNTRY.Continent) and their respective average city populations (CITY.Population) rounded down to the nearest integer.
- MySQl Solution

```sql
select country.continent, round(avg(city.population) -0.5)
from city 
inner join country
on city.countrycode = country.code
group by country.continent;
```