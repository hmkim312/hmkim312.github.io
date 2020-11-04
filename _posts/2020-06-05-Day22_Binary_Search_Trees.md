---
title: Day22 - Binary Search Trees (Python 3)
author: HyunMin Kim
date: 2020-06-05 00:00:00 0000
categories: [Hacker Ranker, 30 Days of Code]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/30-binary-search-trees/problem>{:target="_blank"}

- Objective
    - Today, we're working with Binary Search Trees (BSTs). Check out the Tutorial tab for learning materials and an instructional video!

- Task
    - The height of a binary search tree is the number of edges between the tree's root and its furthest leaf. You are given a pointer, root, pointing to the root of a binary search tree. Complete the getHeight function provided in your editor so that it returns the height of the binary search tree.

- Input Format
    - The locked stub code in your editor reads the following inputs and assembles them into a binary search tree:
    - The first line contains an integer, n, denoting the number of nodes in the tree.
    - Each of the  subsequent lines contains an integer, data, denoting the value of an element that must be added to the BST.

- Output Format
    - The locked stub code in your editor will print the integer returned by your getHeight function denoting the height of the BST.

#### 문제풀이
- 주어진 root값을 시작으로 왼쪽, 오른쪽으로 크고 작은 값을 넣는 이진트리에 관련된 문제였다.
- 사실 이진트리가 무엇인지 제대로 알지못해 문제는 푸는데 조금 어려움이 있었고, 구글링 결과 어느정도 개념은 잡을수 있었다.
- 다행히 hackerrank에서 주어지는 문제는 node의 height를 구하는 문제였으며, 나머지 구조들은 대부분 이미 짜여져있는 상태였다.
- 해당 문제에서 root는 height로 취급하지 않는다.
- root가 none이면 Node 클래스를 생성하여 data를 받고, 이후에 주어지는 인자들이 data보다 작으면 왼쪽으로, 크면 오른쪽으로 가는 형식으로 되어있다.
- getHeight는 왼쪽데이터와 오른쪽 데이터를 비교하여 더 큰쪽에 1씩 더해주어 height를 만들어내는 함수이다.
- 22일차가 되니 점점 알고리즘이 어려워져서 사실 잘 이해가 가지않는 부분이 있어 학습하는데 애를 먹었다.
- 이러한 알고리즘이 있다는것을 알게 되었으니, 반복 학습이 필요할듯 싶다.


```python
class Node:
    def __init__(self,data):
        self.right=self.left=None
        self.data = data
class Solution:
    def insert(self,root,data):
        if root==None:
            return Node(data)
        else:
            if data<=root.data:
                cur=self.insert(root.left,data)
                root.left=cur
            else:
                cur=self.insert(root.right,data)
                root.right=cur
        return root

    def getHeight(self,root):
        if root:
            leftnode = self.getHeight(root.left)
            rightnode = self.getHeight(root.right)
            
            if leftnode > rightnode:
                return leftnode + 1
            else :
                return rightnode + 1
        else:
            return -1
```
