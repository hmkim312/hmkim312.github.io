---
title: Japanese Cities' Attributes
author: HyunMin Kim
date: 2020-11-02 18:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Oracle, Where]
---

# Japanese Cities' Attributes

- URL : <https://www.hackerrank.com/challenges/japanese-cities-attributes/problem>{:target="_blank"}
- Query all attributes of every Japanese city in the CITY table. The COUNTRYCODE for Japan is JPN.
- MySQL, Oracle Solution 

```sql
SELECT *
FROM city
WHERE countrycode = 'JPN';
```