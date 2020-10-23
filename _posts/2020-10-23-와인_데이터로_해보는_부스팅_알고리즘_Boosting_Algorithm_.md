---
title: 와인데이터로 해보는 부스팅 알고리즘(Boosting Algorithm)
author: HyunMin Kim
date: 2020-10-23 09:10:00 0000
categories: [Python, Machine Learning]
tags: [Boosting Algorithm, Voting, Bagging, Ensemble]
---

## 1. 앙상블
---

### 1.1 앙상블이란
- 앙상블은 전통적으로 Voting, Bagging, Boosting, 스태깅으로 나뉨
- 보팅과 배깅은 여러개의 분류기가 투표를 통해 최종 예측 결과를 결정하는 방식임
- 보팅과 배깅의 차이점은 보팅은 각각 다른 분류기, 배깅은 같은 분류기를 사용함
- 대표적인 배깅은 랜덤 포레스트

<br>

### 1.2 Boosting의 개요

<img src="https://user-images.githubusercontent.com/60168331/96965971-ff004080-1547-11eb-9dc0-2fc54590c7e3.png">

- 여러개의 분류기가 순차적으로 학습을 하면서, 앞에서 학습한 분류기가 예측이 틀린 데이터에 대해 다음 분류기가 가중치를 인가해서 학습을 이어 진행하는 방식
- 예측 성능이 뛰어나서 앙상블 학습을 주도함
- 그래디언트 부스트(Gradient Boost), XGBoost, LightGBM 등이 있음 

<br>

### 1.3 배깅과 부스팅의 차이
<img  src="https://user-images.githubusercontent.com/60168331/96966071-248d4a00-1548-11eb-8d06-dc71c81a8fa8.png">

- 배깅 : 한번에 병렬적으로 결과를 얻음
- 부스팅 : 순차적으로 진행이 됨

<br>

### 1.4 Adaboost

<img src="https://user-images.githubusercontent.com/60168331/96966205-5e5e5080-1548-11eb-9725-e520b8928345.png">

- 순차적으로 가중치를 부여해서 최종 결과를 얻음
- AdaBoost는 Decision Tree기반의 알고리즘임
- 여러 Step을 거치며 각 Step에서 틀린 데이터에 가중치를 인가하며 경계선을 결정함
- 마지막으로 앞의 Step들에서 결정한 경계들을 모두 합침

<br>

### 1.5 부스팅 기법
- GBM Gradient Boosting : AdaBoost 기법과 비슷하지면 가중치를 업데이트할때 경사하강법(Gradient Descent)을 사용
- XGBoost : GBM에서 PC의 파워를 효율적으로 사용하기 위해 다양한 기법에 채택되어 빠른 속도와 효율을 가짐
- LigthGBM : XGBoost보다 빠른 속도를 가짐

<br>

### 1.6 Bagging = Bootstrap AGGregatING

<img src="https://user-images.githubusercontent.com/60168331/96966563-fc521b00-1548-11eb-8072-49d9ef24a514.png">

<br>

### 1.7 Bagging과 Boosting의 차이

<img src="https://user-images.githubusercontent.com/60168331/96966707-489d5b00-1549-11eb-8ebd-3aa404054770.png">

<br>

## 2. Wine 데이터로 실습
---

### 2.1 Data load


```python
import pandas as pd

wine_url = 'https://raw.githubusercontent.com/hmkim312/datas/main/wine/wine.csv'

wine = pd.read_csv(wine_url, index_col=0)
wine['taste'] = [1. if grade > 5 else 0. for grade in wine['quality']]

X = wine.drop(['taste','quality'],  axis = 1)
y = wine['taste']
```

- 데이터를 불러오고, quality를 기준으로 taste 컬럼까지 생성

<br>

### 2.2 Scaler 적용 후 데이터 나누기


```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_sc = sc.fit_transform(X)
```


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y, test_size=0.2, random_state=13)
```

<br>

### 2.3 모든 컬럼의 히스토그램 확인


```python
import matplotlib.pyplot as plt
%matplotlib inline

wine.hist(bins = 10, figsize=(24, 24))
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/96967871-155bcb80-154b-11eb-9e52-885b1d1c6fd2.png'>


<br>

### 2.4 Quality 별 다른 특성이 어떤지 확인


```python
colum_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
               'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
               'pH', 'sulphates', 'alcohol']
df_pivot_table = wine.pivot_table(colum_names, ['quality'], aggfunc='median')
df_pivot_table
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
      <th>alcohol</th>
      <th>chlorides</th>
      <th>citric acid</th>
      <th>density</th>
      <th>fixed acidity</th>
      <th>free sulfur dioxide</th>
      <th>pH</th>
      <th>residual sugar</th>
      <th>sulphates</th>
      <th>total sulfur dioxide</th>
      <th>volatile acidity</th>
    </tr>
    <tr>
      <th>quality</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>10.15</td>
      <td>0.0550</td>
      <td>0.33</td>
      <td>0.995900</td>
      <td>7.45</td>
      <td>17.0</td>
      <td>3.245</td>
      <td>3.15</td>
      <td>0.505</td>
      <td>102.5</td>
      <td>0.415</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.00</td>
      <td>0.0505</td>
      <td>0.26</td>
      <td>0.994995</td>
      <td>7.00</td>
      <td>15.0</td>
      <td>3.220</td>
      <td>2.20</td>
      <td>0.485</td>
      <td>102.0</td>
      <td>0.380</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9.60</td>
      <td>0.0530</td>
      <td>0.30</td>
      <td>0.996100</td>
      <td>7.10</td>
      <td>27.0</td>
      <td>3.190</td>
      <td>3.00</td>
      <td>0.500</td>
      <td>127.0</td>
      <td>0.330</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10.50</td>
      <td>0.0460</td>
      <td>0.31</td>
      <td>0.994700</td>
      <td>6.90</td>
      <td>29.0</td>
      <td>3.210</td>
      <td>3.10</td>
      <td>0.510</td>
      <td>117.0</td>
      <td>0.270</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11.40</td>
      <td>0.0390</td>
      <td>0.32</td>
      <td>0.992400</td>
      <td>6.90</td>
      <td>30.0</td>
      <td>3.220</td>
      <td>2.80</td>
      <td>0.520</td>
      <td>114.0</td>
      <td>0.270</td>
    </tr>
    <tr>
      <th>8</th>
      <td>12.00</td>
      <td>0.0370</td>
      <td>0.32</td>
      <td>0.991890</td>
      <td>6.80</td>
      <td>34.0</td>
      <td>3.230</td>
      <td>4.10</td>
      <td>0.480</td>
      <td>118.0</td>
      <td>0.280</td>
    </tr>
    <tr>
      <th>9</th>
      <td>12.50</td>
      <td>0.0310</td>
      <td>0.36</td>
      <td>0.990300</td>
      <td>7.10</td>
      <td>28.0</td>
      <td>3.280</td>
      <td>2.20</td>
      <td>0.460</td>
      <td>119.0</td>
      <td>0.270</td>
    </tr>
  </tbody>
</table>
</div>



- quaity를 기준으로 pivot 테이블을 만들어봄
- free sulfur dioxide가 quality 별로 차이가 나 보인다.

<br>

### 2.5 Quality에 대한 나머지 특성들의 상관관계


```python
corr_matrix = wine.corr()
print(corr_matrix['quality'].sort_values(ascending = False))
```

    quality                 1.000000
    taste                   0.814484
    alcohol                 0.444319
    citric acid             0.085532
    free sulfur dioxide     0.055463
    sulphates               0.038485
    pH                      0.019506
    residual sugar         -0.036980
    total sulfur dioxide   -0.041385
    fixed acidity          -0.076743
    color                  -0.119323
    chlorides              -0.200666
    volatile acidity       -0.265699
    density                -0.305858
    Name: quality, dtype: float64


- quality의 상관관계를 확인해보니, alcohol, free sulfur dioxide가 양의 상과관계를, density가 음의 상관관계를 보인다
- 당연히 quality 기준으로 taste를 만들었으니, 이 둘은 상관관계가 높을수 밖에 없으니 제외함

<br>

### 2.6 Taste 컬럼의 분포


```python
import seaborn as sns

sns.countplot(wine['taste'])
plt.show()
```


<img src = 'https://user-images.githubusercontent.com/60168331/96967880-17be2580-154b-11eb-8c07-059874deebd3.png'>


- Taste 컬럼은 맛있음(1)이 더 많다.

<br>

### 2.7 다양한 모델을 한번에 테스트해보기


```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

models = []
models.append(('RandomForestClassifier',RandomForestClassifier()))
models.append(('DecisionTreeClassifier',DecisionTreeClassifier()))
models.append(('AdaBoostClassifier',AdaBoostClassifier()))
models.append(('GradientBoostingClassifier',GradientBoostingClassifier()))
models.append(('LogisticRegression',LogisticRegression(solver = "liblinear")))
```

- 여러가지 모델을 불러와서 model이라는 리스트에 넣어줌, 하이퍼 파라미터는 설정하지 않음

<br>

### 2.8 결과를 확인


```python
from sklearn.model_selection import KFold, cross_val_score

results  = []
names = []

for name, model in models :
    kfold = KFold(n_splits= 5, random_state=13, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    
    results.append(cv_results)
    names.append(name)
    
    print(name, cv_results.mean(), cv_results.std())
```

    RandomForestClassifier 0.8185420522691939 0.018560021121147078
    DecisionTreeClassifier 0.7498519286295995 0.013712535378522434
    AdaBoostClassifier 0.7533103205745169 0.02644765901536818
    GradientBoostingClassifier 0.7663959428444511 0.021596556352125432
    LogisticRegression 0.7425394240023693 0.015704134753742827


- Kfold를 적용하여 각 모델별로 검증

<br>

### 2.9 Cross-Validation의 결과를 그래프로 보기


```python
fig = plt.figure(figsize=(14,8))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/96967881-17be2580-154b-11eb-84a6-f2f2d1597caf.png'>


- 랜덤포레스트가 좋아보임
- Boxplot으로 보는 이유는 각 데이터의 accuray의 분포와 outlier를 한번에 볼수 있기 때문

<br>

### 2.10 같은 방식으로 test 데이터 대입


```python
from sklearn.metrics import accuracy_score

for name, model in models:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, accuracy_score(y_test, pred))
```

    RandomForestClassifier 0.8392307692307692
    DecisionTreeClassifier 0.7784615384615384
    AdaBoostClassifier 0.7553846153846154
    GradientBoostingClassifier 0.7876923076923077
    LogisticRegression 0.7469230769230769


- 마찬가지로 랜덤포레스트의 결과가 좋음
