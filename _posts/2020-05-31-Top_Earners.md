---
title: Top Earners
author: HyunMin Kim
date: 2020-05-31 20:00:00 0000
categories: [Hacker Ranker, SQL]
tags: [MySQL, Group By, Order By, Limit]
---

# Top Earners

- URL : <https://www.hackerrank.com/challenges/earnings-of-employees/problem>{:target="_blank"}
- We define an employee's total earnings to be their monthly salary * months worked, and the maximum total earnings to be the maximum total earnings for any employee in the Employee table. Write a query to find the maximum total earnings for all employees as well as the total number of employees who have maximum total earnings. Then print these values as 2 space-separated integers.
- MySQL Solution


```sql
select months * salary as earnings, count(*) 
from employee
group by earnings
order by earnings desc limit 1
```