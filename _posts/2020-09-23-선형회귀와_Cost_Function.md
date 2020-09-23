---
title: 선형회귀(Linear Regression)와 Cost Function
author: HyunMin Kim
date: 2020-09-23 12:00:00 0000
categories: [Data Science, Machine Learning]
tags: [Cost Function, Gradient Descent, Linear Regression, Learning Rate]
---


## 1. 기본 개념 잡기

### 1.1 주택 가격 예측
- 주택의 넓이과 가격이라는 데이터가 있고, 주택가격을 예측한다고 했을때, 머신러닝 모델은 어떻게 만들수 있을까?
- 일단 학습데이터 각각에 정답(주택가격)이 있으므로 지도학습이고, 주택가격이 연속된 값이고, 이를 예측하는 것이므로 회귀 문제이다
<br>

### 1.2 선형회귀(Linear Regression)
- 입력변수(특징) x가 1개인 경우 선형회귀문제는 주어진 학습데이터와 가장 잘 맞는 가설(Hypothesis) 함수 h를 찾는 문제가 됨
- <img src = 'https://latex.codecogs.com/gif.latex?h_\theta(x)&space;=&space;\theta_0&space;&plus;&space;\theta_1x'>
<br>

### 1.3 모델을 구성하는 파라미터를 찾는 방법
- 주어진 학습 데이터 x에 대해 정답 y와 예측값 y^의 차이가 최소가 되게 파라미터의 값을 결정 한다.
<br>

### 1.4 직선상에 있지 않은 3개의 점을 직선으로 표현하기
- 3개의 점들과 가장 가깝게 선을 그으면 됨
- 점들과 선의 거리가 제일 가까운위치 (오차가 제일 적은 위치)라는 뜻
- <img src="https://user-images.githubusercontent.com/60168331/93977864-1a93f200-fdb6-11ea-8455-027560bec4dc.png">
<br>

## 2. Cost Function
---
### 2.1 Cost Function이란
- 생성한 모델과 실제 데이터간의 차이
- Cost Function이 작아야, 모델의 성능은 높다
- 위의 예제에서 해당 선이 모델이라고 하였을때 각 3점과의 에러가 가장 작은 선을 찾아야함
<br>

### 2.2 Cost Function 계산
```python
import numpy as np

np.poly1d([2, -1]) ** 2 + np.poly1d([3, -5]) ** 2 + np.poly1d([5, -6]) ** 2
```




    poly1d([ 38, -94,  62])
<br>

### 2.3 최소값 구하기

```python
import scipy as sp
import sympy as sym
th = sym.Symbol('th')
```
<img src = 'https://latex.codecogs.com/gif.latex?\theta'>


```python
diff_th = sym.diff(38 * th ** 2 - 94 * th + 62 ,th)
diff_th
```
<img src = 'https://latex.codecogs.com/gif.latex?76\theta&space;-&space;94'>




```python
sym.solve(diff_th)
```
    [47/38]

<br>

## 3. Cost Function Graph
---

### 3.1 Cost Function - 데이터와 모델이 완전히 일치하면 th = 1

```python
plt.figure(figsize=(12, 8))
plt.scatter([1, 2, 3], [1, 2, 3], marker = 'o', s = 100, c = 'r')
plt.plot(np.linspace(0, 3), np.linspace(0, 3))
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/93983115-20410600-fdbd-11ea-920b-1591aa39a587.png'>

<br>

### 3.2 Cost Function - 데이터와 모델이 반만 일치하면 th = 0.5

```python
plt.figure(figsize=(12, 8))
plt.scatter([1, 2, 3], [1, 2, 3], marker = 'o', s = 100, c = 'r')
plt.plot(np.linspace(0, 3), np.linspace(0, 1.5))
plt.xlim(0,4)
plt.ylim(0,4)
plt.show()
```
<img src = 'https://user-images.githubusercontent.com/60168331/93983120-220ac980-fdbd-11ea-945b-052ffcebe4bc.png'>
<br>

### 3.3 Cost Function - 데이터와 모델이 완전히 안맞을때 th = 0

```python
plt.figure(figsize=(12, 8))
plt.scatter([1, 2, 3], [1, 2, 3], marker = 'o', s = 100, c = 'r')
plt.plot(np.linspace(0, 3), np.linspace(0, 0), color='r')
plt.xlim(0, 4)
plt.ylim(-0.1, 4)
plt.show()
```
<img src = 'https://user-images.githubusercontent.com/60168331/93983124-233bf680-fdbd-11ea-8501-e2adfd8d8e4d.png'>
<br>

### 3.4 cost function에 따른 th의 모양

```python
plt.figure(figsize=(12, 8))
plt.plot([0, 0.5, 1, 1.5, 2], [3, 1, 0, 1, 3], 'o')
plt.plot([0, 0.5, 1, 1.5, 2], [3, 1, 0, 1, 3])
plt.show()
```
<img src = 'https://user-images.githubusercontent.com/60168331/93983127-23d48d00-fdbd-11ea-98fc-159e022e5845.png'>
<br>

### 3.5 실제 데이터는 다르다
- 하지만 실제 데이터는 1차원 데이터가 아닌, 특성들이 많은 다차원 데이터임
- 결국 실제 데이터는 너무 복잡해서 손으로 풀수가 없습니다.
<br>

## 4. 경사하강법(Gradient Descent)
---
### 4.1 경사하강법이란?
- 손쉽게 최소값 찾는 방법
- 랜덤하게 임의의 점을 선택해서 미분(or 편미분)값을 계산해서 업데이트 하여 오른쪽이나, 왼쪽으로 하강하여, 최소점을 찾는 방법
<br>

### 4.1 학습률(Learning Rate)
- 얼마만큼 theta를 갱신할것인지를 설정하는 값
- 학습률이 작다면 
    - 최소값을 찾으러가는 간격이 작게됨
    - 여러번 갱신해야 함
    - 대신 최소값에 잘 도달할수 있음
- 학습률이 높다면
    - 최소값을 찾으러가는 간격이 크게됨
    - 만약 최소값을 찾았다면 갱신횟수는 상대적으로 적을수 있으나, 수렴하지 않고 진동할 수도 있음
<br>

## 5. 다변수 데이터에 대한 회귀 (Multivariate Linear Regression)
---
### 5.1 여러개의 특성
- Multivariate Linear Regression 문제로 일반화 할수 있음
- 회귀문제에서 대부분은 다변수 데이터 일것임
<br>

## 6. 실습) 보스턴 집값으로 예측해 보는 다변수 회귀 문제
---
### 6.1 데이터 불러오기

```python
from sklearn.datasets import load_boston

boston = load_boston()
```
<br>

### 6.2 컬럼 확인 (사실상 왜있는지도 모르는겠는 컬럼이 있기는함..)
- CRIM : 범죄율
- ZN : 25,000 평방 피트를 초과 거주지역 비율
- INDUS : 비소매상업지역 면적 비율
- CHAS : 찰스강의 경계에 위치한 경우는 1, 아니면 0
- NOX : 일산화질소 농도
- RM : 주택당 방수
- AGE : 1940년 이전에 건축된 주택의 비율
- DIS : 직업센터의 거리
- RAD : 방사형 고속도로까지의 거리
- TAX : 재산세율
- PTRATIO : 학생/교사 비율
- B : 인구 중 흑인 비율
- LSTAT : 인구 중 하위 계층 비율


```python
boston.feature_names
```
    array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
           'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7')
<br>

### 6.3 Dataframe 만들기

```python
import pandas as pd

[i for i in boston.feature_names]
boston_pd = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_pd['PRICE'] = boston.target

boston_pd.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>PRICE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>
<br>


### 6.4 Price에 대한 Histogram

```python
# import plotly.express as px

# fig = px.histogram(boston_pd, x= 'PRICE')
# fig.show()

plt.figure(figsize=(12, 8))
plt.hist(data = boston_pd, x = 'PRICE')
plt.show()
```
<img src = 'https://user-images.githubusercontent.com/60168331/93983133-246d2380-fdbd-11ea-974a-56862820fbe7.png'>
<br>

### 6.5 각 특성별 상관계수 확인

```python
import matplotlib.pyplot as plt
import seaborn as sns

corr_mat = boston_pd.corr().round(1)
sns.set(rc={'figure.figsize': (10, 8)})
sns.heatmap(data=corr_mat, annot = True, cmap = 'bwr')
plt.show()
```
<img src = 'https://user-images.githubusercontent.com/60168331/93983135-246d2380-fdbd-11ea-9c38-c8192019c678.png'>

- Price의 rm과 lstat가 상관관계가 높음
- 방의 갯수가 많으면 집이 상대적으로 넓을것이고, 그렇다면 집값이 비싸고, 하위 계층의 비율이 높으면 잘 못사는 지역으로 집값이 싼것으로 생각이 듬

<br>

### 6.6 RM과 LSTAT의 PRICE와이 관계를 더 자세히

```python
sns.set_style('darkgrid')
sns.set(rc={'figure.figsize': (12, 6)})
fig, ax = plt.subplots(ncols=2)
sns.regplot(x = 'RM', y = 'PRICE', data = boston_pd, ax = ax[0])
sns.regplot(x = 'LSTAT', y = 'PRICE', data = boston_pd, ax = ax[1])
plt.show()
```
<img src = 'https://user-images.githubusercontent.com/60168331/93983139-2505ba00-fdbd-11ea-8b72-c334e003cfd4.png'>

- 저소득층 인구가 낮을수록, 방의 갯수가 많을 수록 집값이 높아지는것 처럼 보임
<br>

### 6.7 데이터 분리

```python
from sklearn.model_selection import train_test_split

X = boston_pd.drop('PRICE', axis=1)
y = boston_pd['PRICE']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=13)
```
<br>

### 6.8 LinearRegression

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, y_train)
```
    LinearRegression()
- 회귀 문제로 풀기로했으니, LinearRegression으로 학습
<br>

### 6.9 모델 평가 RMSE로 해보기

```python
import numpy as np
from sklearn.metrics import mean_squared_error

pred_tr = reg.predict(X_train)
pred_test = reg.predict(X_test)
rmse_tr = (np.sqrt(mean_squared_error(y_train, pred_tr)))
rmse_test = (np.sqrt(mean_squared_error(y_test, pred_test)))

print('RMSE of Train Data :', rmse_tr)
print('RMSE of Test Data :', rmse_test)
```

    RMSE of Train Data : 4.642806069019824
    RMSE of Test Data : 4.931352584146711


### 6.10 성능확인

```python
plt.scatter(y_test, pred_test)
plt.xlabel('Actual House Price ($1000)')
plt.ylabel('Predicted Prices')
plt.title('Real vs Predicted')
plt.plot([0, 48], [0, 48], 'r')
plt.show()
```
<img src = 'https://user-images.githubusercontent.com/60168331/93983141-259e5080-fdbd-11ea-8961-f02b8971f349.png'>
<br>

### 6.11 LSTAT를 제외하고 다시 해보기

```python
from sklearn.model_selection import train_test_split

X = boston_pd.drop(['PRICE', 'LSTAT'], axis=1)
y = boston_pd['PRICE']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=13)

reg = LinearRegression()
reg.fit(X_train, y_train)
```
    LinearRegression()
<br>

### 6.12 성능은 나빠짐

```python
import numpy as np
from sklearn.metrics import mean_squared_error

pred_tr = reg.predict(X_train)
pred_test = reg.predict(X_test)
rmse_tr = (np.sqrt(mean_squared_error(y_train, pred_tr)))
rmse_test = (np.sqrt(mean_squared_error(y_test, pred_test)))

print('RMSE of Train Data :', rmse_tr)
print('RMSE of Test Data :', rmse_test)
```

    RMSE of Train Data : 5.165137874244864
    RMSE of Test Data : 5.295595032597162

- 당연히 상관관계가 높은 특성을 뺐으니..
- 하지만 RMSE가 5.1, 5.2가 나오는게 얼마나 좋은것인지 판단이 안섬
<br>

### 6.13 그래프로 보기

```python
plt.scatter(y_test, pred_test)
plt.xlabel('Actual House Price ($1000)')
plt.ylabel('Predicted Prices')
plt.title('Real vs Predicted')
plt.plot([0, 48], [0, 48], 'r')
plt.show()
```
<img src = 'https://user-images.githubusercontent.com/60168331/93983147-2636e700-fdbd-11ea-8459-babe088099dd.png'>

- 어떤 특성을 넣고 뺴고 하는 고민은 항상 있을듯 싶다.
