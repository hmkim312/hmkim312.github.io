---
title: African Cities
author: HyunMin Kim
date: 2020-05-31 10:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL]
---

# African Cities

- URL : <https://www.hackerrank.com/challenges/african-cities/problem>{:target="blank"}
- Given the CITY and COUNTRY tables, query the names of all cities where the CONTINENT is 'Africa'.
- Note: CITY.CountryCode and COUNTRY.Code are matching key columns.
- Mysql solution

```sql
select city.name
from city, country
where countrycode = code and continent = 'africa';
```