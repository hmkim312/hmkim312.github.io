---
title: Select By ID
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Oracle]
---


# Select By ID

- URL : <https://www.hackerrank.com/challenges/select-by-id/problem>{:target="_blank"}
- Query all columns for a city in CITY with the ID 1661.
- MySQL, Oracle Solution


```sql
SELECT * 
FROM city
WHERE id = 1661;
```