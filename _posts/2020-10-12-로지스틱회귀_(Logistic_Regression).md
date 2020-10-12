---
title: 로지스틱 회귀 (Logistic Regression)
author: HyunMin Kim
date: 2020-10-12 21:10:00 0000
categories: [Data Science, Machine Learning]
tags: [Logistic Regression, Wine Data, PIMA Data, Important Feature]
---

## 1. 로지스틱 회귀 (Logistic Regression)
---
### 1.1 분류? 회귀? 악성 종양을 찾는 문제

<img src="https://user-images.githubusercontent.com/60168331/95749252-42e87f80-0cd6-11eb-8caf-c55d4ad6a549.png">

- 0.5보다 크거나 같으면 1(악성)으로 예측
- 0.5보다 작으면 0(양성)으로 예측

<br>

### 1.2 Linear Regression을 분류 문제에 적용?

<img src="https://user-images.githubusercontent.com/60168331/95749252-42e87f80-0cd6-11eb-8caf-c55d4ad6a549.png">
- 회귀는 0보다 작은수, 1보다 큰수가 나오므로 그대로 분류문제에 사용할수 없음

<br>

### 1.3 모델 재 설정
- 분류 문제는 0 또는 1로 예측해야 하나 Linear Regression을 그대로 적용하면 예측값은 0보다 작거나 1보다 큰 값을 가지게 됨
- 예측값이 항상 0에서 1 사이의 값을 가지게 하도록 hypothesis 함수를 수정한다

<br>

### 1.4 Logistic Function 그래프로 보기

```python
import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-10, 10, 0.01)
g = 1 / (1 + np.exp(-z))

plt.plot(z, g)
plt.show()
```

<img src ='https://user-images.githubusercontent.com/60168331/95752946-cf497100-0cdb-11eb-8e87-855bb5db87c5.png'>

<br>

### 1.5 디테일하게

```python
plt.figure(figsize=(12, 8))
ax = plt.gca()

ax.plot(z, g)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('center')
ax.spines['top'].set_color('none')

plt.grid()
plt.show()
```

<img src ='https://user-images.githubusercontent.com/60168331/95752949-d07a9e00-0cdb-11eb-8a27-69564cbd5c05.png'>

<br>

### 1.6 Logistic Reg, Cost Function의 그래프

```python
h = np.arange(0.01, 1, 0.01)

C0 = -np.log(1 -h)
C1 = -np.log(h)

plt.figure(figsize=(12,8))
plt.plot(h, C0, label = 'y=0')
plt.plot(h, C1, label = 'y=1')
plt.legend()

plt.show()
```

<img src ='https://user-images.githubusercontent.com/60168331/95752950-d1133480-0cdb-11eb-822c-aea081b0e720.png'>

<br>

## 2. Wine 데이터로 실습
---
### 2.1 데이터 로드

```python
import pandas as pd

wine_url = 'https://raw.githubusercontent.com/hmkim312/datas/main/wine/wine.csv'

wine = pd.read_csv(wine_url, index_col=0)
wine.head()
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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



- 실습용 자료를 git에 올려놓았음

### 2.2 맛등급 넣기


```python
wine['taste']= [1. if grade > 5 else 0. for grade in wine['quality']]

X = wine.drop(['taste', 'quality'], axis = 1)
y = wine['taste']
```

- quality가 0보다 작으면 0, 크면 1로 하는 taste라는 컬럼을 생성


<br>

### 2.3 데이터 분리

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 13)
```

<br>

### 2.4 로지스틱 회귀 (LogisticRegression)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression(solver= 'liblinear', random_state=13)
lr.fit(X_train, y_train)

y_pred_tr = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

print('Train Acc : ', accuracy_score(y_train, y_pred_tr))
print('Test Acc : ', accuracy_score(y_test, y_pred_test))
```

    Train Acc :  0.7427361939580527
    Test Acc :  0.7438461538461538

- 로지스틱 회귀 적용
    - penalty : 패널티를 부여할 때 사용할 기준을 결정
    - dual : bool type, dual formulation or primal formulation
    - tol : 중지 기준에 대한 허용 오차 값
    - C : 규칙 강도의 역수 값
    - fit_intercept : bool type, 의사 결정 기능에 상수를 추가할지 여부
    - class_weight : 클래스에 대한 가중치 값
    - solver : 최적화에 사용할 알고리즘 (newton-cg, lbfgs, liblinear, sag, saga)
    - max_iter : solver가 수렴하게 만드는 최대 반복 값
    - multi_class : ovr, multinomial


<br>

### 2.5 파이프라인 구축

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

estimators = [('scaler', StandardScaler()),
              ('clf', LogisticRegression(solver='liblinear', random_state=13))]

pipe = Pipeline(estimators)
```

- Standartscaler를 적용하여 파이프라인을 구축함 

<br>

### 2.6 학습

```python
pipe.fit(X_train, y_train)
```
    Pipeline(steps=[('scaler', StandardScaler()),
                    ('clf',
                     LogisticRegression(random_state=13, solver='liblinear'))])


<br>

### 2.7 결과 확인

```python
y_pred_tr = pipe.predict(X_train)
y_pred_test = pipe.predict(X_test)

print('Train Acc : ', accuracy_score(y_train, y_pred_tr))
print('Test Acc : ', accuracy_score(y_test, y_pred_test))
```

    Train Acc :  0.7444679622859341
    Test Acc :  0.7469230769230769

- 스케일러를 적용한 결과 아주 조금 Accuracy가 올랐음


<br>

### 2.8 Decision Tree와의 비교

```python
from sklearn.tree import DecisionTreeClassifier

wine_tree = DecisionTreeClassifier(max_depth= 2, random_state= 13)
wine_tree.fit(X_train, y_train)

models = {'logistic Regression' : pipe, 'Decision Tree' : wine_tree}
```

<br>

### 2.9 ROC 그래프를 이용한 모델간 비교


```python
from sklearn.metrics import roc_curve

plt.figure(figsize=(10, 8))
plt.plot([0,1], [0,1])
for model_name, model in models.items():
    pred = model.predict_proba(X_test)[:, -1]
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    plt.plot(fpr, tpr, label = model_name)
    
plt.grid()
plt.legend()
plt.show()
```

<img src ='https://user-images.githubusercontent.com/60168331/95752951-d1abcb00-0cdb-11eb-907c-c22792e8b608.png'>

- Roc 커브를 보았을때 해당 로지스틱 회귀가 조금더 성능이 괜찮은것으로 보인다

<br>

## 3. PIMA 인디언 당뇨병 예측
---
### 3.1 PIMA 인디언 문제?
- 1950년대 까지 PIMA인디언은 당뇨가 없었음
- PIMA인디언은 강가 수렵을 하던 소수 인디언이나, 미국 정부에 의해 강제 이후 후 식량을 배급 받았음
- 하자만 20세기말 인구의 50%가 당뇨에 걸림

<br>

### 3.1 데이터 로드

```python
import pandas as pd

PIMA_url = 'https://raw.githubusercontent.com/hmkim312/datas/main/pima/diabetes.csv'
PIMA = pd.read_csv(PIMA_url)
PIMA.head()
```

<br>

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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

<br>

- 원본 데이터는 kaggle에 있으며, 해당 데이터를 깃헙링크를 올림
- 컬럼의 의미
    - Pregnancies : 임신 횟수
    - Glucose : 포도당 부하 검사 수치     
    - BloodPressure : 혈압
    - SkinThickness : 팔 삼두근 뒤쪽의 피하지방 측정값       
    - Insulin : 혈청 인슐린
    - BMI : 체질량 지수       
    - DiabetesPedigreeFunction : 당뇨 내력 가중치 값
    - Age : 나이
    - Outcome : 당뇨의 유무  

<br>

### 3.2 데이터 확인

```python
PIMA.info()
```
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Pregnancies               768 non-null    int64  
     1   Glucose                   768 non-null    int64  
     2   BloodPressure             768 non-null    int64  
     3   SkinThickness             768 non-null    int64  
     4   Insulin                   768 non-null    int64  
     5   BMI                       768 non-null    float64
     6   DiabetesPedigreeFunction  768 non-null    float64
     7   Age                       768 non-null    int64  
     8   Outcome                   768 non-null    int64  
    dtypes: float64(2), int64(7)
    memory usage: 54.1 KB


<br>

### 3.3 Float으로 변환

```python
PIMA = PIMA.astype('float')
PIMA.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Pregnancies               768 non-null    float64
     1   Glucose                   768 non-null    float64
     2   BloodPressure             768 non-null    float64
     3   SkinThickness             768 non-null    float64
     4   Insulin                   768 non-null    float64
     5   BMI                       768 non-null    float64
     6   DiabetesPedigreeFunction  768 non-null    float64
     7   Age                       768 non-null    float64
     8   Outcome                   768 non-null    float64
    dtypes: float64(9)
    memory usage: 54.1 KB

<br>

### 3.4 상관관계 확인

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
sns.heatmap(PIMA.corr(), cmap= 'YlGnBu')
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/95752954-d2446180-0cdb-11eb-836d-ff0cf8a25799.png' >

- Outcome과 비교하여 상관 관계가 낮은 컬럼들이 있음

<br>

### 3.5 데이터가 0인 outlier가 있음

```python
(PIMA == 0).astype('int').sum()
```
    Pregnancies                 111
    Glucose                       5
    BloodPressure                35
    SkinThickness               227
    Insulin                     374
    BMI                          11
    DiabetesPedigreeFunction      0
    Age                           0
    Outcome                     500
    dtype: int64

- 0이라는 숫자가 혈압에 있다면 문제가 있는것으로 파악

<br>

### 3.6 0을 평균값으로 대체 


```python
zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
PIMA[zero_features] = PIMA[zero_features].replace(0, PIMA[zero_features].mean())
(PIMA ==0).astype('int').sum()
```
    Pregnancies                 111
    Glucose                       0
    BloodPressure                 0
    SkinThickness                 0
    Insulin                     374
    BMI                           0
    DiabetesPedigreeFunction      0
    Age                           0
    Outcome                     500
    dtype: int64

- 의학적인 지삭과 PIMA 인디언에 대한 정보는 없지만, 일단 0을 평균값으로 대체함

<br>

### 3.7 데이터 분리

```python
from sklearn.model_selection import train_test_split

X = PIMA.drop(['Outcome'], axis=1)
y = PIMA['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=13, stratify=y)
```

<br>

### 3.8 pipeline 만들기

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

estimators = [('scaler', StandardScaler()),
              ('clf', LogisticRegression(solver='liblinear', random_state=13))]

pipe_lr = Pipeline(estimators)
pipe_lr.fit(X_train, y_train)
pred = pipe_lr.predict(X_test)
```

- 로지스틱회귀와, 스탠다드 스케일러를 적용한 파이프라인을 구축함

<br>

### 3.9 수치 확인

```python
from sklearn.metrics import (accuracy_score, recall_score, f1_score, precision_score, roc_auc_score)

print('Accuracy : ', accuracy_score(y_test, pred))
print('Reacll : ', recall_score(y_test, pred))
print('Precision : ', precision_score(y_test, pred))
print('AUC score: ', roc_auc_score(y_test, pred))
print('f1 score : ', f1_score(y_test, pred))
```
    Accuracy :  0.7727272727272727
    Reacll :  0.6111111111111112
    Precision :  0.7021276595744681
    AUC score:  0.7355555555555556
    f1 score :  0.6534653465346535


- 사실상 해당 수치가 상대적 의미를 가질수 없어서, 이 수치 자체를 평가 할 수 없음

<br>

### 3.10 다변수 방정식의 각 계수값을 확인 가능

```python
coeff = list(pipe_lr['clf'].coef_[0])
labels = list(X_train.columns)
coeff
```
    [0.3542658884412649,
     1.201424442503758,
     -0.15840135536286715,
     0.033946577129299486,
     -0.16286471953988116,
     0.6204045219895111,
     0.3666935579557874,
     0.17195965447035108]

<br>

### 3.11 중요한 feature그리기

```python
features = pd.DataFrame({'Features': labels, 'importance': coeff})
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features['importance'].plot(kind='barh', figsize=(
    11, 6), color=features['positive'].map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')
plt.show()
```
<img src = 'https://user-images.githubusercontent.com/60168331/95752957-d2dcf800-0cdb-11eb-9df1-779d25b26c80.png'>

- 포도당, BMI 등은 당뇨에 영향을 미치는 정도가 높다
- 혈압은 예측에 부정적 영향을 준다
- 연령이 BMI보다 출력 변수와 더 관련되어 있었지만, 모델은 BMI와 Glucose에 더 의존한다,