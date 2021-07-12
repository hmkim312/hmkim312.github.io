---
title: Sorting Algorithm
author: HyunMin Kim
date: 2021-07-12 00:00:00 0000
categories: [Data Science, Algorithm]
tags: [Sorting, Time Complexity]
---

### 출처
- 해당 post는 <https://www.youtube.com/watch?v=Bor_CRWEIXo>{:target="_blank"} 영상을 보고 정리한것 입니다.

### 0. Big O
- 알고리즘의 성능을 이해하기 쉽고 효율적으로 작성하는 방법
- 하지만, 항상 Big O가 모든 알고리즘을 완벽하게 설명하는것은 아님
- 같은 Big O의 성능을 가지고 있더라도 알고리즘의 퍼포먼스가 다른것을 설명하고자 한다.

### 1. Sorting
- 사전처럼 무언가를 기준으로 정렬하는것을 **Sorting (정렬)** 이라 한다.
- 이진검색 처럼 빠른 알고리즘을 사용하려면 정렬은 기본으로 되어있어야 한다.
- 이번 포스트에서 알아볼 정렬 알고리즘은 아래의 3가지 이다.
    - Bubble Sort (버블 정렬)
    - Selection Sort (선택 정렬)
    - Insertion Sort (삽입 정렬)
    - 위의 3가지 정렬 알고리즘은 가장 빠른 알고리즘은 아니나, 사람이 정렬하는것과 유사한 방법을 가지며, 시간 복잡도를 계산하는것도 쉽다.

### 2. Bubble Sort (버블 정렬)

```python
[5, 2, 6, 3, 1, 4]
[2, 5, 6, 3, 1, 4] # cycle : 1 swaps : 1, compare : 1, num : 2, 5 
[2, 5, 6, 3, 1, 4] # cycle : 1 swaps : 1, compare : 2, num : 5, 6
[2, 5, 3, 6, 1, 4] # cycle : 1 swaps : 2, compaer : 3, num : 6, 3
[2, 5, 3, 1, 6, 4] # cycle : 1 swaps : 3, compaer : 4, num : 6, 1
[2, 5, 3, 1, 4, 6] # cycle : 1 swaps : 4, compaer : 5, num : 4, 6

[2, 5, 3, 1, 4, 6] # cycle : 2 swaps : 4, compaer : 6, num : 2, 5
[2, 3, 5, 1, 4, 6] # cycle : 2 swaps : 5, compaer : 7, num : 5, 3
[2, 3, 1, 5, 4, 6] # cycle : 2 swaps : 6, compaer : 8, num : 5, 1
[2, 3, 1, 4, 5, 6] # cycle : 2 swaps : 7, compaer : 9, num : 5, 4
[2, 3, 1, 4, 5, 6] # cycle : 2 swaps : 7, compaer : 10, num : 5, 6

[2, 3, 1, 4, 5, 6] # cycle : 3 swaps : 7, compaer : 11, num : 2, 3
[2, 1, 3, 4, 5, 6] # cycle : 3 swaps : 8, compaer : 12, num : 3, 1
[2, 1, 3, 4, 5, 6] # cycle : 3 swaps : 8, compaer : 13, num : 3, 4
[2, 1, 3, 4, 5, 6] # cycle : 3 swaps : 8, compaer : 14, num : 4, 5
[2, 1, 3, 4, 5, 6] # cycle : 3 swaps : 8, compaer : 15, num : 5, 6

[1, 2, 3, 4, 5, 6] # cycle : 4 swaps : 9, compaer : 16, num : 2, 1

```

- 배열의 처음 2개 원소를 선택 후 2개의 원소를 비교 후 왼쪽에는 작은 원소, 오른쪽에는 큰 원소로 바꾼다.
- 이후 다음 2개의 원소를 선택해서 동일한 과정을 끝까지 거친다.
- 위의 [5, 2, 6, 3, 1, 4]의 배열에서 보면 먼저 맨앞의 2개 원소 5,2를 비교하여 2를 5앞으로 보낸다.
- 그 다음 5,6을 비교하였고 5가 더 작기 때문에 그대로 둔다
- 다음 6과 3을 비교하고 3이 더 작기 때문에 왼쪽으로 바꾼다.
- 위의 과정을 총 4번의 사이클을 거치면 [1, 2, 3, 4, 5, 6]의 배열로 변경되게 된다.

<img src = "https://user-images.githubusercontent.com/60168331/125214881-63407b80-e2f4-11eb-9534-fca8ebc785fd.png">

- 버블정렬의 시간 복잡도는 최악의 경우 모든 사이클마다, 모든 아이템을 비교하고 교환해야 하므로 O(n^2)이다.
- 시간 복잡도가 오래걸리므로 좋은 알고리즘이 아니다.

### 3. Selection Sort (선택 정렬)

```python
[5, 2, 6, 3, 1, 4] # cycle : 1, swaps : 0,  compare : 1, num : 5, 2, small elements : 5, index : 0
[5, 2, 6, 3, 1, 4] # cycle : 1, swaps : 0,  compare : 2, num : 2, 6, small elements : 2, index : 1
[5, 2, 6, 3, 1, 4] # cycle : 1, swaps : 0,  compare : 3, num : 6, 3, small elements : 2, index : 1
[5, 2, 6, 3, 1, 4] # cycle : 1, swaps : 0,  compare : 4, num : 3, 1, small elements : 1, index : 4
[5, 2, 6, 3, 1, 4] # cycle : 1, swaps : 0,  compare : 5, num : 1, 4, small elements : 1, index : 4
[1, 2, 6, 3, 5, 4] # cycle : 1, swaps : 1,  compare : 5, num : None, small elements : 1, index : 0

[1, 2, 6, 3, 5, 4] # cycle : 2, swaps : 1,  compare : 6, num : 2, 6, small elements : 2, index : 1
[1, 2, 6, 3, 5, 4] # cycle : 2, swaps : 1,  compare : 7, num : 6, 3, small elements : 2, index : 1
[1, 2, 6, 3, 5, 4] # cycle : 2, swaps : 1,  compare : 8, num : 3, 5, small elements : 2, index : 1
[1, 2, 6, 3, 5, 4] # cycle : 2, swaps : 1,  compare : 9, num : 5, 4, small elements : 2, index : 1
[1, 2, 6, 3, 5, 4] # cycle : 2, swaps : 1,  compare : 9, num : None, small elements : 2, index : 1

[1, 2, 6, 3, 5, 4] # cycle : 3, swaps : 1,  compare : 10, num : 6, 3, small elements : 3, index : 3
[1, 2, 6, 3, 5, 4] # cycle : 3, swaps : 1,  compare : 11, num : 3, 5, small elements : 3, index : 3
[1, 2, 6, 3, 5, 4] # cycle : 3, swaps : 1,  compare : 12, num : 5, 4, small elements : 3, index : 3
[1, 2, 3, 6, 5, 4] # cycle : 3, swaps : 2,  compare : 12, num : None, small elements : 3, index : 2

[1, 2, 3, 6, 5, 4] # cycle : 4, swaps : 2,  compare : 13, num : 6, 5, small elements : 5, index : 4
[1, 2, 3, 6, 5, 4] # cycle : 4, swaps : 2,  compare : 14, num : 5, 4, small elements : 4, index : 5
[1, 2, 3, 4, 5, 6] # cycle : 4, swaps : 3,  compare : 14, num : None, small elements : 4, index : 3
```

- 전체 원소중에서 가장 작은 원소의 위치를 변수에 저장하고, 더 작은 원소가 나올때까지 다음 원소와 **비교**만 함
- 배열을 끝까지 다 돌면 가장 작은 원소를 맨 앞의 원소와 위치를 변경 한다.
- 위의 사이클을 배열이 정렬될때까지 진행한다.

<img src = "https://user-images.githubusercontent.com/60168331/125214881-63407b80-e2f4-11eb-9534-fca8ebc785fd.png">

- 선택 정렬도 버블 정렬과 마찬가지로 원소를 비교해야하는 것은 N-1이지만 사이클마다 원소를 교환하는것은 1번만 하면 되기 때문에 사실 버블 정렬보다 선택 정렬이 훨씬 빠르다.
- 하지만 선택 정렬을 시간복잡도로 표현하면 마찬가지로 O(N^2)이다.

### 4. Insertion Sort (삽입 정렬)
```python
[5, 2, 6, 3, 1, 4]
[2, 5, 6, 3, 1, 4] # cycle : 1, swaps : 1, compare : 1, start index : 1, num : 2, 5

[2, 5, 6, 3, 1, 4] # cycle : 2, swaps : 1, compare : 2, start index : 2, num : 5, 6

[2, 5, 3, 6, 1, 4] # cycle : 3, swaps : 2, compare : 3, start index : 3, num : 6, 3
[2, 3, 5, 6, 1, 4] # cycle : 3, swaps : 3, compare : 4, start index : 3, num : 5, 3

[2, 3, 5, 1, 6, 4] # cycle : 4, swaps : 4, compare : 5, start index : 4, num : 6, 1
[2, 3, 1, 5, 6, 4] # cycle : 4, swaps : 5, compare : 6, start index : 4, num : 5, 1
[2, 1, 3, 5, 6, 4] # cycle : 4, swaps : 6, compare : 7, start index : 4, num : 3, 1
[1, 2, 3, 5, 6, 4] # cycle : 4, swaps : 7, compare : 8, start index : 4, num : 2, 1

[1, 2, 3, 5, 4, 6] # cycle : 5, swaps : 8, compare : 9, start index : 5, num : 6, 4
[1, 2, 3, 4, 5, 6] # cycle : 5, swaps : 9, compare : 10, start index : 5, num : 5, 4
```

- 삽입 정렬은 1번 인덱스에서 시작해서 왼쪽의 원소와 비교하여 더 작으면 왼쪽의 원소와 교환하는것으로 한다.
- 만일 왼쪽 원소가 더 작으면 해당 사이클은 종료 된다.
- 삽입 정렬은 필요한 원소만 스캔하기 때문에 선택 정렬보다 빠르다.

<img src = "https://user-images.githubusercontent.com/60168331/125214881-63407b80-e2f4-11eb-9534-fca8ebc785fd.png">

- 삽입 정렬이 선택 정렬과 버블 정렬보다 당연히 빠르지만, 시간 복잡도는 마찬가지로 O(n^2)이다.

### 5. Summary

- 위에서 살펴본 모든 정렬들의 속도는 분명 차이가 나지만, 시간 복잡도는 마찬가지로 O(n^2)으로 동일하다. 이유가 무엇일까?

<img src = "https://user-images.githubusercontent.com/60168331/125216324-cf24e300-e2f8-11eb-8e54-5945acb4bbab.png">

- 삽입 정렬과 선택 정렬 모두 평균적으로는 O(n^2)의 시간 복잡도를 가지고 있고, 삽입 정렬의 경우 가장 좋은 경우에만 O(n)의 시간 복잡도를 가진다.
- 중요한것은 동일한 시간 복잡도를 가진다고 하여도, 그 사이에서도 더 빠른 알고리즘이 있을수 있고, 최고, 평균, 최악의 경우에는 시간 복잡도가 달라지기도 한다는 것이다.
