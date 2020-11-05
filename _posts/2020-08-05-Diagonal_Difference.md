---
title: Diagonal Difference (Python 3)
author: HyunMin Kim
date: 2020-08-05 00:00:00 0000
categories: [Hacker Ranker, Problem Solving]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/diagonal-difference/problem>{:target="_blank"}


- Given a square matrix, calculate the absolute difference between the sums of its diagonals.

- For example, the square matrix arr is shown below:
```
1 2 3
4 5 6
9 8 9 
```
- The left-to-right diagonal = 1 + 5 + 9 = 15. The right to left diagonal = 3 + 5 + 9 = 17. Their absolute difference is |15 - 17| = 2.

- Function description
    - Complete the diagonalDifference function in the editor below.
    - diagonalDifference takes the following parameter:
    - int arr[n][m]: an array of integers
    
- Return
    - int: the absolute diagonal difference

- Input Format
    - The first line contains a single integer, n, the number of rows and columns in the square matrix arr.
    - Each of the next n lines describes a row, arr[i], and consists of n space-separated integers arr[i][j].

- Constraints
    - -100 <= arr[i][j] <= 100

- Output Format
    - Return the absolute difference between the sums of the matrix's two diagonals as a single integer.

#### 문제풀이
- 정방행렬이 주어지고, 해당 행렬을 대각선 방향으로 다 더한 원소들끼리의 차를 절대값으로 뱉는 함수를 작성
- 정방행렬이니 arr[1][1], arr[2][2] 처럼 같은 숫자의 위치를 더하면됨, 반대쪽 행렬은 arr[0][-1], arr[1][-2] 처럼 뒤의 숫자만 -1씩 더해짐
- 위의 설명을 for문을 이용하여 코드로 작성 


```python
def diagonalDifference(arr):
    a = 0
    b = 0
    for i in range(len(arr)):
        a += arr[i][i]
        b += arr[i][-(i+1)]

    return abs(a - b)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    result = diagonalDifference(arr)

    fptr.write(str(result) + '\n')

    fptr.close()
```




    15