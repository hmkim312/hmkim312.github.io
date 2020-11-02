---
title: Asian Population
author: HyunMin Kim
date: 2020-05-31 11:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL]
---

# Asian Population

- URL : <https://www.hackerrank.com/challenges/asian-population/problem>{:target="blank"}
- Given the CITY and COUNTRY tables, query the sum of the populations of all cities where the CONTINENT is 'Asia'.
- Note: CITY.CountryCode and COUNTRY.Code are matching key columns.
- MySQL Solution

```sql
select sum(city.population)
from city
join country on city.countrycode = country.code
where country.continent = 'asia';
```