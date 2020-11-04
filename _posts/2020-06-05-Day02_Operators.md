---
title: Day02 - Operators (Python 3)
author: HyunMin Kim
date: 2020-06-05 00:00:00 0000
categories: [Hacker Ranker, 30 Days of Code]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/30-operators/problem>{:target="_blank"}

- Objective
    - In this challenge, you'll work with arithmetic operators. Check out the Tutorial tab for learning materials and an instructional video!

- Task
    - Given the meal price (base cost of a meal), tip percent (the percentage of the meal price being added as tip), and tax percent (the percentage of the meal price being added as tax) for a meal, find and print the meal's total cost.

- Note
    - Be sure to use precise values for your calculations, or you may end up with an incorrectly rounded result!

- Input Format
    - There are 3 lines of numeric input:
    - The first line has a double,  mealcost (the cost of the meal before tax and tip).
    - The second line has an integer, tippercent (the percentage of mealcost being added as tip).
    - The third line has an integer, taxpercent (the percentage of mealcost being added as tax).

- Output Format
    - Print the total meal cost, where totalcost is the rounded integer result of the entire bill (mealcost with added tax and tip).


```python
import math
import os
import random
import re
import sys

# Complete the solve function below.
def solve(meal_cost, tip_percent, tax_percent):
    totalcost = round((meal_cost) + (meal_cost *(tip_percent / 100)) + (meal_cost *(tax_percent / 100)))
    print(totalcost)
```


```python
meal_cost = 12
tip_percent = 20
tax_percent = 8
solve(meal_cost, tip_percent, tax_percent)
```

    15