---
title: Japanese Cities' Names
author: HyunMin Kim
date: 2020-05-31 19:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Oracle, Where]
---

# Japanese Cities' Names

- URL : <https://www.hackerrank.com/challenges/japanese-cities-name/problem>{:target="_blank"}
- Query the names of all the Japanese cities in the CITY table. The COUNTRYCODE for Japan is JPN.
- MySQL, Oracle Solution

```sql
select name
from city
where countrycode = 'JPN';
```