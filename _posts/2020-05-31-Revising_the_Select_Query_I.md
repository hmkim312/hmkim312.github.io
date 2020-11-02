---
title: Revising the Select Query I
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Oracle, Where]
---


# Revising the Select Query I

- URL : <https://www.hackerrank.com/challenges/revising-the-select-query/problem>{:target="_blank"}
- Query all columns for all American cities in the CITY table with populations larger than 100000. The CountryCode for America is USA.
- MySQL, Oracle Solution

```sql
SELECT *
FROM city
WHERE population > 100000 and countrycode = 'USA';
```