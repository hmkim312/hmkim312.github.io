---
title: Binary Search
author: HyunMin Kim
date: 2021-06-28 00:00:00 0000
categories: [Data Science, Algorithm]
tags: [Binary Search, Time Complexity, Array]
---

### 출처
- 해당 post는 <https://www.youtube.com/watch?v=WjIlVlmmNqs>{:target="_blank"} 영상을 보고 정리한것 입니다.

### Binary Search & Linear Search
- 두 알고리즘 모두 검색을 최대한 빠르게하는 Search 알고리즘이다.

### Linear Search

```python
[32, 2, 20, 1, 7, 8, 11, 8, 28, 40]
```

- Linear Search는 위처럼 10개의 숫자가 있는 배열에서 숫자 7을 찾는다면 33부터 **순차적으로** 하나씩 배열을 Search해 나간다.
- 알고리즘의 Step으로 생각한다면 총 5번의 Step이 소요된것이다.
- Linear Search는 찾는 숫자가 배열의 제일 뒤에 있거나, 아에 없는 경우에는 배열을 처음부터 끝까지 모두 훑어봐야 한다. 이때 Step이 배열의 길이만큼 되는 것이 단점이고 이를 Linear time Complexity라고 한다.

### Linear Time Complexity
- Input이 많으면 수행하는 Time 역시 선형적으로 증가하는것
- 이를 해결하기 위해 Binary Search를 사용한다.

### Binary Search
- Linear Search는 어느 배열에서나 사용 가능한것과 달리 **Binary Search는 정렬된 배열에서만 사용가능하다.**
- Sorted Array는 배열이 순서대로 정렬되어 있는것을 말한다. 예를 들자면 작은수 부터 큰수까지 정렬된 배열이다.
- Sorted Array는 Search에서는 Linear Array보다 엄청 빠르지만, 새로운 원소를 추가하는데는 새로운 원소가 배열의 중간에 들어가기 때문에 Linear Array보다 시간이 오래 걸린다.
- 즉, 정리하자면 Sorted Array는 Binary Search 알고리즘 때문에 Search는 빠르지만 add는 시간이 오래걸린다.
- 하지만 Sorted Array에서 Search는 속도가 굉장히 빠르다.


- 일단 여기서 이야기하는 Binary는 0과 1을 의미하는것이 아닌, 하나를 2개로 쪼개는 것을 의미한다.
- 그런 의미로 생각해본다면 Binary Search는 0부터 시작하는것이 아닌 배열의 가운데를 비교한다.
- 배열의 가운데가 target보다 작다면 왼쪽을, 크다면 오른쪽의 배열을 확인하면서 target을 찾는다.

```python
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

- 위처럼 1부터 10까지 Sorted Array가 있을때 9을 Binary Search 알고리즘을 사용해서 찾는다고 하면 어떻게 될까?
- 1 step) 우선 배열의 중앙의 값인 5를 target인 9과 비교한다. 9은 5보다 크므로 왼쪽의 배열은 더 이상 신경쓰지않는다.
- 2 step) 오른쪽 배열 [6,7,8,9,10]의 중앙값인 8을 target인 9과 비교한다. 이번에는 target이 중앙값 8보다 크므로 왼쪽 배열은 신경쓰지 않는다.
- 3 step) 오른쪽 배열 [9,10]에서 다시 중앙값 9를 찾았다. 이것은 target과 동일하므로 Search가 종료된다.
- 10개의 원소가 있는 배열에서 target 9를 찾을때 소요된 Step은 3번이다.

```python
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
```
- 이번에는 위의 배열에서 13을 찾아보도록 하자.
- 1 Step) 중앙값 11은 13보다 작으므로 오른쪽 [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]의 배열만 남는다.
- 2 Step) 다시 남은 배열에서 중앙값 15와 target 13을 비교한다. 이번에는 중앙값이 target보다 크므로 왼쪽의 [11, 12, 13, 14] 배열만 남는다.
- 3 Step) 남은 배열의 중앙값 12와 target을 비교하여 오른쪽 배열 [13, 14]가 남는다.
- 4 Step) [13, 14] 배열의 중앙값 13이 target과 동일하므로 Search는 종료된다.
- 앞보다 배열은 2배가 되었지만, Step은 1개만 늘어나게 되었다. 만일 Linear Search였다면 최대 10번의 Step이 더 소요될수도 있었다.
- 만일 1만개의 원소가 있는 Array라면 Linear Search는 최악의 경우 1만 Step이 필요하지만, Binary Search의 경우 14 Step만 필요하다. 이는 엄청난 차이이다.
- 이처럼 Binary Search는 Linear Search보다 Search에 있어서는 굉장히 빠른 성능을 보여준다. 더욱이 배열의 크기가 커질수록 그 차이는 극대화 된다.
- Binary Search를 사용하기 위해서는 Sort를 해야하며 Sort를 하게 된다면 Add에 있어서는 속도가 떨어진다.
- 이러한 알고리즘의 장점과 단점을 잘 파악 후 사용해야 효율적인 데이터 구조를 얻게 될것 이다.