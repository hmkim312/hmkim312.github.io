---
title: 머신러닝 기초 - 의사결정나무(2)
author: HyunMin Kim
date: 2020-09-20 22:30:00 0000
categories: [Datascience, Machine Learning]
tags: [Sklearn, Decision Tree, Stratify]
---

## 1. 머신러닝
### 1.1 지도학습이란
- 학습 대상이 되는 데이터에 정답(label)을 붙여서 학습 시키고
- 모델을 얻어서 완전히 새로운 데이터에 모델을 사용해서 '답'을 얻고자 하는것

### 1.2 데이터의 분리 (훈련(train)/ 검증(validation)/ 평가(test))
- 전체 데이터에서 train / test 로 데이터를 분리하고
- train 데이터를 validation 데이터로 또다시 분리하는 것
- 과적합을 피하기 위해 실행

### 1.3 과적합이란
- 모델이 train데이터에 너무 맞춰저서 학습하게 되는것
- 머신러닝 기초 - 의사결정나무(1)에서 결정 경계의 선이 복잡했던것을 떠올리면 된다.

## 2. iris data로 실습
### 2.1 데이터 로드 후 분리 
```python
from sklearn.datasets import load_iris
iris = load_iris()

from sklearn.model_selection import train_test_split
features = iris.data[:, 2:]
labels = iris.target

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, random_state = 13, stratify = labels)
```
- 종속변수(y, target)값의 비율이 동일하게 하기 위해서는 stratify  옵션을 주면 됨
- 위의 코드는 8:2로 train, test 데이터가 분리됨

### 2.2 train으로만 Decssion Tree Model 생성
```python
from sklearn.tree import DecisionTreeClassifier

iris_tree = DecisionTreeClassifier(max_depth=2, random_state=13)
iris_tree.fit(X_train, y_train)
```
- 학습할 때 마다 일관성을 위해 random_state만 고정
- 모델을 단순화시키기 위해 max_depth를 조정

### 2.3 train data 의 Accuracy 확인
```python
from sklearn.metrics import accuracy_score

y_pred_tr = iris_tree.predict(X_train)
accuracy_score(y_train, y_pred_tr)
```
```python
0.95
```
- iris데이터는 단순해서 acc가 높게 나타남

### 2.4 모델 확인 with Graphviz
```python
from graphviz import Source
from sklearn.tree import export_graphviz

Source(export_graphviz(iris_tree, feature_names=['length', 'width'],
                       class_names=iris.target_names,
                       rounded=True, filled=True))
```
<img src ='https://user-images.githubusercontent.com/60168331/93713820-6289fd80-fb99-11ea-856d-f6b830414e4d.png'>

### 2.5 훈련 데이터에 대한 결정경계 확인
```python
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

plt.figure(figsize = (12, 8))
plot_decision_regions(X= X_train, y= y_train, clf = iris_tree, legend= 2)
plt.show()
```
<img src = 'https://user-images.githubusercontent.com/60168331/93713836-84838000-fb99-11ea-9f11-d484a8ce2cdf.png'>


### 2.6 테스트 데이터에 대한 accuracy

```python
y_pred_test = iris_tree.predict(X_test)
accuracy_score(y_test, y_pred_test)
```
    0.9666666666666667


### 2.7 테스트 데이터에 대한 경계

```python
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

plt.figure(figsize = (12, 8))
plot_decision_regions(X= X_test, y= y_test, clf = iris_tree, legend= 2)
plt.show()
```
<img src = 'https://user-images.githubusercontent.com/60168331/93713837-85b4ad00-fb99-11ea-8fe1-6d33c6c126d2.png'>


### 2.8 전체 데이터에서 관찰하기

```python
scatter_highlight_kwargs = {'s': 150, 'label': 'Test data', 'alpha': 0.9}
scatter_kwargs = {'s' : 120, 'edgecolor': None, 'alpha': 0.7}

plt.figure(figsize=(12, 8))
plot_decision_regions(X=features, y=labels, X_highlight=X_test, clf=iris_tree, legend=2,
                      scatter_highlight_kwargs=scatter_highlight_kwargs,
                      scatter_kwargs=scatter_kwargs,
                      contourf_kwargs={'alpha': 0.2})
plt.show()
```
<img src = 'https://user-images.githubusercontent.com/60168331/93713839-86e5da00-fb99-11ea-8183-757a49020aeb.png'>


### 2.9 feature를 네 개

```python
features = iris.data
labels = iris.target

X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    test_size=0.2,
                                                    stratify=labels,
                                                    random_state=13)

iris_tree = DecisionTreeClassifier(max_depth=2, random_state=13)
iris_tree.fit(X_train, y_train)

Source(export_graphviz(iris_tree, feature_names=iris.feature_names, class_names=iris.target_names,
                       rounded=True, filled=True))
```

<img src = 'https://user-images.githubusercontent.com/60168331/93713840-88170700-fb99-11ea-9ede-872547d1c210.png'>


### 2.10 모델 사용법

```python
# 길가다가 주운 iris가 sepal과 petal의 length, width가 각각 [4.3, 2. , 1.2, 1.0]이라면
test_data = [[4.3, 2.0, 1.2, 1.0]]
iris_tree.predict_proba(test_data)
```
    array([[0.        , 0.97222222, 0.02777778]])

```python
# 각 클래스별 확률이 아니라 범주값을 바로 알고 싶다면
iris.target_names
```
    array(['setosa', 'versicolor', 'virginica'], dtype='<U10')

```python
test_data = [[4.3, 2.0, 1.2, 1.0]]
iris.target_names[iris_tree.predict(test_data)]
```
    array(['versicolor'], dtype='<U10')


### 1.17 주요특성 확인하기
-  Black box / White box 모델
- Tree계열 알고리즘은 특성을 파악하는데 장점을 가진다

```python
iris_tree.feature_importances_
```
    array([0.        , 0.        , 0.42189781, 0.57810219])


```python
iris_clf_model = dict(zip(iris.feature_names, iris_tree.feature_importances_))
iris_clf_model
```

    {'sepal length (cm)': 0.0,
     'sepal width (cm)': 0.0,
     'petal length (cm)': 0.421897810218978,
     'petal width (cm)': 0.578102189781022}