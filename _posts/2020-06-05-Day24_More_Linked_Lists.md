---
title: Day24 - More Linked Lists (Python 3)
author: HyunMin Kim
date: 2020-06-05 00:00:00 0000
categories: [Hacker Ranker, 30 Days of Code]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/30-linked-list-deletion/problem>{:target="_blank"}

- Objective
    - Check out the Tutorial tab for learning materials and an instructional video!

- Task
    - A Node class is provided for you in the editor. A Node object has an integer data field, data, and a Node instance pointer, next, pointing to another node(i.e.: the next node in a list).

    - A removeDuplicates function is declared in your editor, which takes a pointer to the head node of a linked list as a parameter. Complete removeDuplicates so that it deletes any duplicate nodes from the list and returns the head of the updated list.

- Note: The head pointer may be null, indicating that the list is empty. Be sure to reset your next pointer when performing deletions to avoid breaking the list.

- Input Format
    - You do not need to read any input from stdin. The following input is handled by the locked stub code and passed to the removeDuplicates function:
    - The first line contains an integer, N, the number of nodes to be inserted.
    - The N subsequent lines each contain an integer describing the data value of a node being inserted at the list's tail.

- Constraints
    - The data elements of the linked list argument will always be in non-decreasing order.

- Output Format
    - Your removeDuplicates function should return the head of the updated linked list. The locked stub code in your editor will print the returned list to stdout.


```python
def removeDuplicates(self,head):
    #Write your code here
    if head is None:
        return None
    ls = [head.data]
    node = head
    while node.next is not None:
        if node.next.data not in ls:
            node = node.next
        else :
            snode = node.next
            node.next = snode.next
            del snode
        ls.append(node.data)
    return head
```
