---
title: Japan Population
author: HyunMin Kim
date: 2020-05-31 18:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Oracle, Where]
---


# Japan Population

- URL : <https://www.hackerrank.com/challenges/japan-population/problem>{:target="_blank"}
- Query the sum of the populations for all Japanese cities in CITY. The COUNTRYCODE for Japan is JPN.
- MySQL, Oracle Solution

```sql
select sum(population)
from city
where countrycode = 'JPN';
```