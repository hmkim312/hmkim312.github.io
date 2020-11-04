---
title: Day12 - Inheritance (Python 3)
author: HyunMin Kim
date: 2020-06-05 00:00:00 0000
categories: [Hacker Ranker, 30 Days of Code]
tags: [Class]
---

- URL : <https://www.hackerrank.com/challenges/30-inheritance/problem>{:target="_blank"}

- Objective
    - Today, we're delving into Inheritance. Check out the attached tutorial for learning materials and an instructional video!

- Task
    - You are given two classes, Person and Student, where Person is the base class and Student is the derived class. Completed code for Person and a declaration for Student are provided for you in the editor. Observe that Student inherits all the properties of Person.

    - Complete the Student class by writing the following:

    - A Student class constructor, which has  parameters:
        - A string, first name.
        - A string, last name.
        - An integer, id.
        - An integer array (or vector) of test scores, scores.
- A char calculate() method that calculates a Student object's average and returns the grade character representative of their calculated average:
    - Grading Scale
        - O : 90 <= a <= 100
        - E : 80 <= a < 90
        - A : 70 <= a < 80
        - P : 55 <= a < 70
        - D : 40 <= a < 55
        - T : a < 40

- Input Format
    - The locked stub code in your editor calls your Student class constructor and passes it the necessary arguments. It also calls the calculate method (which takes no arguments).
    - You are not responsible for reading the following input from stdin:
      The first line contains first name, last name, and id, respectively. The second line contains the number of test scores. The third line of space-separated integers describes scores.

- Constraints
    - 1 <= |first name|, |last name| <= 10
    - |id| = 7
    - 0 <= score, average <= 100

- Output Format
    - This is handled by the locked stub code in your editor. Your output will be correct if your Student class constructor and calculate() method are properly implemented.

#### 문제풀이
- 설명은 긴데.. 그냥 Person 클래스를 상속받고, 생성자 함수에서는 score를 추가하여 준다
- score를 평균을 낸 뒤에, 해당 평균이 어느 구간에 속하면 O, E, A, P, T, D를 리턴하게 만들어주면 됨


```python
class Person:
	def __init__(self, firstName, lastName, idNumber):
		self.firstName = firstName
		self.lastName = lastName
		self.idNumber = idNumber
	def printPerson(self):
		print("Name:", self.lastName + ",", self.firstName)
		print("ID:", self.idNumber)

class Student(Person):
    #   Class Constructor
    #   
    #   Parameters:
    #   firstName - A string denoting the Person's first name.
    #   lastName - A string denoting the Person's last name.
    #   id - An integer denoting the Person's ID number.
    #   scores - An array of integers denoting the Person's test scores.
    #
    # Write your constructor here
    def __init__(self, firstName, lastName, idNumber, scores):
        self.firstName = firstName
        self.lastName = lastName
        self.idNumber = idNumber
        self.scores = scores
    
    #   Function Name: calculate
    #   Return: A character denoting the grade.
    #
    # Write your function here
    def calculate(self):
        avg = sum(self.scores) / len(self.scores)
        if avg >= 90:
            return 'O'
        elif avg >= 80:
            return 'E'
        elif avg >= 70:
            return 'A'
        elif avg >= 55:
            return 'P'
        elif avg >= 40:
            return 'D'
        else :
            return 'T'
```

     1
