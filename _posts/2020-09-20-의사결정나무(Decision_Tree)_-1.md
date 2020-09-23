---
title: 의사결정나무(Decision Tree) -1
author: HyunMin Kim
date: 2020-09-20 21:30:00 0000
categories: [Data Science, Machine Learning]
tags: [Sklearn, Decision Tree, Graphviz, Mlxtend]
---


## 1. 머신 러닝이란?
### 1.1 머신러닝의 교과서적 정의
- 명시적으로 프로그래밍을 하지 않고 컴퓨터에 학습할 수 있는 능력을 부여여한 학문
- 과거의 데이터로부터 얻은 경험이 쌓여 감에 따라 주어진 태스크의 성능이 점점 좋아질 때 컴퓨터 프로그램은 경험으로부터 학습한다.
- Machine이 주어진 데이터를 통해 규칙을 찾는것

## 2. 의사결정나무(Decision Tree)
### 2.1 의사결정나무(Decision Tree) 란?
- 이진분류와 회귀에 모두 사용하능
- 정답을 알려주는 지도학습 모델 
- 특정 기준에 따라 데이터를 구분하는 모델
- Node : 특정 기준이나 결과를 담은 네모 상자 
- Root Node : 가장 처음에 있는 노드
- Terminal Node(leaf Node) : 맨 마지막 노드

### 2.2 의사결정나무(Decision Tree)의 분할 기준(Split Criterion)
#### 2.2.1 정보 획득(Information Gain)
- 정보의 가치를 반환하는데 발생하는 사전의 확률이 작을 수록 정보의 가치를 커짐
- 정보 이득이란 어떤 속성을 선택함으로 인해서 데이터를 더 잘 구분하게 되는것

#### 2.2.2 엔트로피 개념
- 열역학의 용어로 물질의 열적 상태를 나타내는 물리량 단위중 하나, 무질서의 정도를 나타냄
- 얼마만큼의 정보를 담고 있는가? 또한, 무질서도를 의미, 불확실성을 나타내기도 함
- <img src = 'https://latex.codecogs.com/gif.latex?-p_ilog2p_i'>
- <img src = 'https://user-images.githubusercontent.com/60168331/93713044-8139c580-fb94-11ea-9818-5b769009c3ac.png'>
- p는 해당 데이터가 해당 클래스에 속할 확률
- 엔트로피는 이 확률들의 합이다
- <img src = 'https://latex.codecogs.com/gif.latex?\textup{Entroyp}&space;=&space;\sum_{k=1}^{m}&space;-p_ilog2p_i'>

#### 2.2.3 엔트로피 연습
```python
# 분할 전 엔트로피
-(10/16) * np.log2(10/16) - 6/16 * np.log2(6/16)
```
- 0.95443

```python
# 분할 후 엔트로피
0.5 * (-(7/8) * np.log2(7/8) - 1/8 * np.log2(1/8)) + \
    0.5 * (-(3/8) * np.log2(3/8) - 5/8 * np.log2(5/8))
```
- 0.7899
- 분할 후 엔트로피가 내려갔으니, 분할하는 것이 좋다

#### 2.2.4 지니계수
- Gine index 혹은 불순도율
- <img src = 'https://latex.codecogs.com/gif.latex?\textup{Gini}&space;=&space;\sum_{k=1}^{d}R_i\left&space;\{&space;1&space;-&space;\sum_{m}^{k=1}p^2_i_k&space;\right&space;\}'>
```python
# 분할 전 지니계수
1 - (6/16) ** 2 - (10/16) ** 2
```
- 0.468
```python
# 분할 후 지니계수
0.5 * (1 - (7/8) ** 2 - (1/8) ** 2) + 0.5 * (1 - (3/8) ** 2 - (5/8) ** 2)
```
- 0.34375
- 분할 후 지니계수가 낮으므로, 분할하는 것이 좋다

#### 2.2.5 지니계수 or 엔트로피

- 어떻게 나눌것인지는 지니계수든, 엔트로피든 알고리즘을 동원해서 코드로 작성
- 최근에는 Frame Work로 작성하게 됨

## 3. Scikit Learn
### 3.1 Scikt Learn이란?
- 2007년 구글 썸머코드에서 처음구현
- 현재 파이썬에서 가장 유명한 기계학습 오픈 소스 라이브러리

### 3.2 Scikit Learn을 이용한 DecsionTree 구현 with iris data
```python
from sklearn.tree import DecisionTreeClassifier
iris_tree = DecisionTreeClassifier()
iris_tree.fit(iris.data[:,2:], iris.target)
```

### 3.2 Accuracy 확인
```python
from sklearn.metrics import accuracy_score
y_pred_tr = iris_tree.predict(iris.data[:, 2:])
accuracy_score(iris.target, y_pred_tr)
```
- 0.9933333333333333

## 4. Tree model Visualization 
### 4.1 Graphviz 설치
- Graphviz install for Mac user
  - brew install graphviz
  - pip install graphviz

### 4.2 Graphviz 결과
<img src = 'https://user-images.githubusercontent.com/60168331/93713101-d675d700-fb94-11ea-8ece-ad8316f717df.png'>

### 4.3 mlxtend 설치
- pip install mlxtend

### 4.4 iris 품종을 분류하는 결정나무 모델이 어떻게 데이터를 분류하였는가?
```python
from mlxtend.plotting import plot_decision_regions

plt.figure(figsize=(14, 8))
plot_decision_regions(X = iris.data[:, 2:], y = iris.target, clf = iris_tree, legend= 2)
plt.show()
```
<img src = 'https://user-images.githubusercontent.com/60168331/93713048-826af280-fb94-11ea-88ed-266090e05050.png'>

## 5. 머신러닝의 일반적 절차에 대해서
### 5.1 Accuracy가 높다고 다 믿을수 있을까?
- 경계면은 올바른 것일까?
- 결과는 내가 가진 데이터를 벗어나서 일반화 할수 있을까?
- 어차피 얻은 데이터는 유한하고 내가 얻은 데이터를 이용하여 일반화를 추구
- 복잡한 경계면은 모델의 성능을 결국 나쁘게 만듬(과적합)