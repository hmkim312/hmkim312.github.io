---
title: The Blunder
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Round, Avg, Replace]
---

# The Blunder

- URL : <https://www.hackerrank.com/challenges/the-blunder/problem>{:target="_blank"}
- Samantha was tasked with calculating the average monthly salaries for all employees in the EMPLOYEES table, but did not realize her keyboard's 0 key was broken until after completing the calculation. She wants your help finding the difference between her miscalculation (using salaries with any zeroes removed), and the actual average salary. Write a query calculating the amount of error, and round it up to the next integer.
- MySQL Solution

```sql
select round(avg(salary)) - round(avg(replace(salary, 0, '')))
from EMPLOYEES;
```