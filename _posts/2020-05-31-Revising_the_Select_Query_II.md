---
title: Revising the Select Query II
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Oracle, Where]
---

# Revising the Select Query II

- URL : <https://www.hackerrank.com/challenges/revising-the-select-query-2/problem>{:target="_blank"}
- Query the NAME field for all American cities in the CITY table with populations larger than 120000. The CountryCode for America is USA.
- MySQL, Oracle Solution

```sql
SELECT name
FROM city
WHERE population > 120000 and countrycode = 'USA';
```