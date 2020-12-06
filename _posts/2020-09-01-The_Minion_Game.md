---
title: The Minion Game (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/the-minion-game/problem>{:target="_blank"}


- Kevin and Stuart want to play the 'The Minion Game'.

- Game Rules
    - Both players are given the same string, S.
    - Both players have to make substrings using the letters of the string S.
    - Stuart has to make words starting with consonants.
    - Kevin has to make words starting with vowels.
    - The game ends when both players have made all possible substrings.

- Scoring
    - A player gets +1 point for each occurrence of the substring in the string S.

- For Example:
    - String S = BANANA
    - Kevin's vowel beginning word = ANA
    - Here, ANA occurs twice in BANANA. Hence, Kevin will get 2 Points.

- Input Format
    - A single line of input containing the string S.
    - Note: The string  will contain only uppercase letters: [A-Z].

- Constraints
    - 0 <len(s) <= 10^6


- Output Format
    - Print one line: the name of the winner and their score separated by a space.
    - If the game is a draw, print Draw.

#### 문제풀이
- Stuart는 자음으로 시작하는 단어를 만들고, Kevin은 모음으로 시작하는 단어를 만들어서, 그 갯수만큼을 점수로 가져감
- 모음 혹은 자음이니, 모음(AEIOU)로 시작하는 단어와 아닌 단어만 구분하면됨(자음을 따로 구성할 필요없음)
- len(string) - idx를 하는 이유는 모음이나 자음으로 시작하는 단어 수가 필요하기 때문.
- 단어 "BANANA"의 경우, 첫 번째 모음 'A'는 위치 1, len("BANANA") = 6에서 발생하므로이 문자 'A'로 시작하는 6-1 = 5 개의 하위 문자열이 있습니다. ['A', ' AN ', 'ANA ', 'ANAN ', 'ANANA '], 단어 끝에 도달 할 때까지 특정 문자'A '에 하나의 추가 문자를 추가하게됨
- 위의 내용을 기반으로 for문을 돌려서 점수를 만들어냄
- 솔직히 많이 어려웠음 특히 len(s) - i 하는 부분은 어떻게 해야하나 고민하다가 결국 구글링으로 찾아냄


```python
def minion_game(s):
    Kevin = 0
    Stuart = 0

    for idx, c in enumerate(s):
        if c in "AEIOU":
            Kevin += len(s) - idx
        else:
            Stuart += len(s) - idx

    if Kevin > Stuart:
        print('Kevin', Kevin)
    elif Kevin < Stuart:
        print('Stuart', Stuart)
    else :
        print('Draw')
```


```python
s = 'BANANA'
minion_game(s)
```

    Stuart 12

