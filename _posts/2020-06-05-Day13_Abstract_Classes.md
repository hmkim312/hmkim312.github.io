---
title: Day13 - Abstract Classes (Python 3)
author: HyunMin Kim
date: 2020-06-05 00:00:00 0000
categories: [Hacker Ranker, 30 Days of Code]
tags: [Class]
---

- URL : <https://www.hackerrank.com/challenges/30-abstract-classes/problem>{:target="_blank"}

- Objective
    - Today, we're taking what we learned yesterday about Inheritance and extending it to Abstract Classes. Because this is a very specific Object-Oriented concept, submissions are limited to the few languages that use this construct. Check out the Tutorial tab for learning materials and an instructional video!

- Task
    - Given a Book class and a Solution class, write a MyBook class that does the following:
    - Inherits from Book
    - Has a parameterized constructor taking these 3 parameters:
        - string title
        - string author
        - int price
    - Implements the Book class' abstract display() method so it prints these 3 lines:
        - title, a space, and then the current instance's title.
        - author, a space, and then the current instance's author.
        - price, a space, and then the current instance's price.
- Note: Because these classes are being written in the same file, you must not use an access modifier (e.g.:pubilc ) when declaring MyBook or your code will not execute.

- Input Format
    - You are not responsible for reading any input from stdin. The Solution class creates a Book object and calls the MyBook class constructor (passing it the necessary arguments). It then calls the display method on the Book object.

- Output Format
    - The void display() method should print and label the respective title, author, and price of the MyBook object's instance (with each value on its own line) like so:

        - Title: \$title
        - Author: \$author
        - Price: \$price

- Note: The & is prepended to variable names to indicate they are placeholders for variables.

#### 문제풀이
- Book이라는 클래스를 상속받을떄 추상클래스를 생성하는것
- 아래에서는 Book에서는 타이틀, 작가를 받는데, MyBook는 타이틀, 작가, 가격까지 받음
- 이후 display라는 메서드로 입력받은 내용을 출력해주면 됨


```python
from abc import ABCMeta, abstractmethod
class Book(object, metaclass=ABCMeta):
    def __init__(self,title,author):
        self.title=title
        self.author=author   
    @abstractmethod
    def display(): pass

#Write MyBook class
class MyBook(Book):
    def __init__(self, title, author, price):
        self.title = title
        self.author = author
        self.price = price

    def display(self):
        print('Title:', self.title)
        print('Author:', self.author)
        print('Price:',  self.price)
```


```python
title=input()
author=input()
price=int(input())
new_novel=MyBook(title,author,price)
new_novel.display()
```

     해리포터
     존레논
     30000


    Title: 해리포터
    Author: 존레논
    Price: 30000