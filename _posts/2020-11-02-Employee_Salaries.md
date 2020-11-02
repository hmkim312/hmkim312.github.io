---
title: Employee Salaries
author: HyunMin Kim
date: 2020-11-02 17:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Order By, Where]
---

# Employee Salaries

- URL : <https://www.hackerrank.com/challenges/salary-of-employees/problem>{:target="_blank"}
- Write a query that prints a list of employee names (i.e.: the name attribute) for employees in Employee having a salary greater than $2000 per month who have been employees for less than 10 months. Sort your result by ascending employee_id.
- MySQL Solution

```sql
select name
from Employee
where salary >= 2000 and months < 10
order by employee_id asc
```