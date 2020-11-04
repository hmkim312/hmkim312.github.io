---
title: Day15 - Linked List (Python 3)
author: HyunMin Kim
date: 2020-06-05 00:00:00 0000
categories: [Hacker Ranker, 30 Days of Code]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/30-linked-list/problem>{:target="_blank"}


- Objective
    - Today we're working with Linked Lists. Check out the Tutorial tab for learning materials and an instructional video!

    - A Node class is provided for you in the editor. A Node object has an integer data field, data, and a Node instance pointer, next, pointing to another node (i.e.: the next node in a list).

    - A Node insert function is also declared in your editor. It has two parameters: a pointer, head, pointing to the first node of a linked list, and an integer data value that must be added to the end of the list as a new Node object.

- Task
    - Complete the insert function in your editor so that it creates a new Node (pass data as the Node constructor argument) and inserts it at the tail of the linked list referenced by the head parameter. Once the new node is added, return the reference to the head node.

- Note: If the head argument passed to the insert function is null, then the initial list is empty.

- Input Format
    - The insert function has 2 parameters: a pointer to a Node named head, and an integer value, data.
    - The constructor for Node has 1 parameter: an integer value for the data field.
    - You do not need to read anything from stdin.

- Output Format
    - Your insert function should return a reference to the head node of the linked list.


```python
class Node:
    def __init__(self,data):
        self.data = data
        self.next = None 
class Solution: 
    def display(self,head):
        current = head
        while current:
            print(current.data,end=' ')
            current = current.next

    def insert(self,head,data):
        newnode = Node(data)
        if not head:
            return newnode
        else:
            current = head
            while(current.next):
                current = current.next
            current.next = newnode
        return head
```
