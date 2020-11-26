---
title: Day04 - Class vs Instance (Python 3)
author: HyunMin Kim
date: 2020-06-05 00:00:00 0000
categories: [Hacker Ranker, 30 Days of Code]
tags: [Class]
---

- URL : <https://www.hackerrank.com/challenges/30-class-vs-instance/problem>{:target="_blank"}

- Objective
    - In this challenge, we're going to learn about the difference between a class and an instance; because this is an Object Oriented concept, it's only enabled in certain languages. Check out the Tutorial tab for learning materials and an instructional video!

- Task
    - Write a Person class with an instance variable, age , and a constructor that takes an integer, initialAge, as a parameter. The constructor must assign initialAge to age after confirming the argument passed as initialAge is not negative; if a negative argument is passed as initialAge, the constructor should set age to 0 and print Age is not valid, setting age to 0.. In addition, you must write the following instance methods:

    - 1. yearPasses() should increase the age instance variable by 1.
    - 2. amIOld() should perform the following conditional actions:
        - If age < 13, print You are young..
        - If  age >= 13 and age < 18, print You are a teenager..
        - Otherwise, print You are old..
- To help you learn by example and complete this challenge, much of the code is provided for you, but you'll be writing everything in the future. The code that creates each instance of your Person class is in the main method. Don't worry if you don't understand it all quite yet!

- Note
    - Do not remove or alter the stub code in the editor.

- Input Format

    - Input is handled for you by the stub code in the editor.

    - The first line contains an integer, T (the number of test cases), and the T subsequent lines each contain an integer denoting the age of a Person instance.

- Constraints
    - 1 <= T <= 4
    - -5 <= age <= 30
- Output Format
    - Complete the method definitions provided in the editor so they meet the specifications outlined above; the code to test your work is already in the editor. If your methods are implemented correctly, each test case will print  or  lines (depending on whether or not a valid initialAge was passed to the constructor).


```python
class Person:
    def __init__(self,initialAge):
        # Add some more code to run some checks on initialAge
        if initialAge > 0:
            self.age = initialAge
        else :
            print('Age is not valid, setting age to 0.')
            self.age = 0
        
    def amIOld(self):
        # Do some computations in here and print out the correct statement to the console
        if self.age < 13:
            print('You are young.')
        elif self.age < 18:
            print('You are a teenager.')
        else:
            print('You are old.')
    def yearPasses(self):
        # Increment the age of the person in here
        self.age += 1
```