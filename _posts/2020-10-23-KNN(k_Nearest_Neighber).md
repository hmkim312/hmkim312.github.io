---
title: kNN(k Nearest Neighber)
author: HyunMin Kim
date: 2020-10-23 13:10:00 0000
categories: [Data Science, Machine Learning]
tags: [kNN, Confusion Matrix]
---


# 

## 1. kNN
---
### 1.1 kNN이란?
- 새로운 데이터가 있을 때, 기존 데이터의 그룹 중 어떤 그룹에 속하는지를 분류하는 문제
- k는 몇번째 가까운 데이터까지볼 것인가를 정하는 수치를 뜻함
- k = 5로 설정하면 5번째까지 가까운 데이터라는 뜻으로, k값에 따라 결과값이 바뀔수 있음
- 각 데이터의 거리는 유클리드를 사용함
- 데이터의 단위에 따라 바뀔수 있으므로, 수치 표준화가 필요함

<br>

### 1.2 장단점
- 실시간 예측을 위한 학습이 필요하지 않아 속도가 빠르다
- 고차원 데이터에는 적합하지 않다

<br>

## 2. Iris데이터로 실습
---
### 2.1 Data load 및 분리


```python
from sklearn.datasets import load_iris

iris = load_iris()
```


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.2, random_state = 13, stratify = iris.target)
```

- 훈련과 학습 데이터의 분리는 sklearn의 train_test_split을 사용하였고, 옵션 중 stratiy는 y값의 비율을 훈련과 학습 데이터의 비율에 맞게 조절해준다.

<br>

### 2.2 kNN 적용


```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors= 5)
knn.fit(X_train, y_train)
```




    KNeighborsClassifier()



- k = 5로 설정(n_neighbors = 5)로 하여 학습을 시킴

<br>

### 2.3 Accuracy 확인


```python
from sklearn.metrics import accuracy_score

pred = knn.predict(X_test)
print(accuracy_score(y_test, pred))
```

    0.9666666666666667


- 생각보다 높은 accuracy가 나온다
- 애초에 아이리스 데이터가 단순하기 때문이기도 하다.

<br>

### 2.4 Confusion matrix 확인


```python
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, pred))
print()
print(classification_report(y_test, pred))
```

    [[10  0  0]
     [ 0  9  1]
     [ 0  0 10]]
    
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        10
               1       1.00      0.90      0.95        10
               2       0.91      1.00      0.95        10
    
        accuracy                           0.97        30
       macro avg       0.97      0.97      0.97        30
    weighted avg       0.97      0.97      0.97        30
    


- 0,1,2는 iris의 종류인 'setosa', 'versicolor', 'virginica'이다
- 2 = virginica 인데, 이것의 precision이 조금 낮은것으로 보인다
