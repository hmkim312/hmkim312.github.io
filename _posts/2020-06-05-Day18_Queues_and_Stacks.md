---
title: Day18 - Queues and Stacks (Python 3)
author: HyunMin Kim
date: 2020-06-05 00:00:00 0000
categories: [Hacker Ranker, 30 Days of Code]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/30-queues-stacks/problem>{:target="_blank"}

- Welcome to Day 18! Today we're learning about Stacks and Queues. Check out the Tutorial tab for learning materials and an instructional video!
- A palindrome is a word, phrase, number, or other sequence of characters which reads the same backwards and forwards. Can you determine if a given string, S, is a palindrome?

- To solve this challenge, we must first take each character in S, enqueue it in a queue, and also push that same character onto a stack. Once that's done, we must dequeue the first character from the queue and pop the top character off the stack, then compare the two characters to see if they are the same; as long as the characters match, we continue dequeueing, popping, and comparing each character until our containers are empty (a non-match means S isn't a palindrome).

- Write the following declarations and implementations:

    - Two instance variables: one for your stack, and one for your queue.
    - A void pushCharacter(char ch) method that pushes a character onto a stack.
    - A void enqueueCharacter(char ch) method that enqueues a character in the queue instance variable.
    - A char popCharacter() method that pops and returns the character at the top of the stack instance variable.
    - A char dequeueCharacter() method that dequeues and returns the first character in the queue instance variable.
    
- Input Format
    - You do not need to read anything from stdin. The locked stub code in your editor reads a single line containing string S. It then calls the methods specified above to pass each character to your instance variables.

- Constraints
    - S is composed of lowercase English letters.

- Output Format
    - You are not responsible for printing any output to stdout.
    - If your code is correctly written and S is a palindrome, the locked stub code will print The word, s, is a palindrome; otherwise, it will print The word, s, is not a palindrome

#### 문제풀이
- 거꾸로해도 똑같은 단어인지 확인하는 것.
- 거꾸로해도 같은 단어인지를 Stack과 queue를 이용하여 확인


```python
import sys

class Solution:
    # Write your code here
    stack = []
    queue = []

    def pushCharacter(self, s):
        self.stack.append(s)

    def popCharacter(self):
        return self.stack.pop()

    def enqueueCharacter(self, s):
        self.queue.append(s)

    def dequeueCharacter(self):
        char = self.queue[0]
        del self.queue[0]
        return char
```


```python
s = 'racecar'
obj=Solution()   

l=len(s)
# push/enqueue all the characters of string s to stack
for i in range(l):
    obj.pushCharacter(s[i])
    obj.enqueueCharacter(s[i])
    
isPalindrome=True
'''
pop the top character from stack
dequeue the first character from queue
compare both the characters
''' 
for i in range(l // 2):
    if obj.popCharacter()!=obj.dequeueCharacter():
        isPalindrome=False
        break
#finally print whether string s is palindrome or not.
if isPalindrome:
    print("The word, "+s+", is a palindrome.")
else:
    print("The word, "+s+", is not a palindrome.")    
```

    The word, racecar, is a palindrome.