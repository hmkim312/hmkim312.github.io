---
title: HAR 데이터로 해보는 GBM, XGBoost, LightGBM
author: HyunMin Kim
date: 2020-10-23 10:10:00 0000
categories: [Data Science, Machine Learning]
tags: [Boosting Algorithm, GBM, XGBoost, LightGBM]
---

## 1. GBM - Gradient Boosting Machine
---

### 1.1 GBM
- 부스팅 알고리즘은 여러 개의 약한 학습기를 순차적으로 학습 예측하면서 잘못 예측한 데이터에 가중치를 부여해서 오류를 개선해 나가는 방식
- GBM은 가중치를 업데이트할때 경사 하강법을 사용하는것이 큰 차이

<br>

### 1.2 HAR 데이터로 실습


```python
import pandas as pd
import matplotlib.pyplot as plt


url = 'https://raw.githubusercontent.com/hmkim312/datas/main/HAR/features.txt'

feature_name_df = pd.read_csv(url, sep = '\s+', header = None, names = ['column_index','column_name'])
feature_name = feature_name_df.iloc[:,1].values.tolist()
X_train = pd.read_csv('https://raw.githubusercontent.com/hmkim312/datas/main/HAR/X_train.txt', sep = '\s+',  header = None)
X_test = pd.read_csv('https://raw.githubusercontent.com/hmkim312/datas/main/HAR/X_test.txt', sep = '\s+',  header = None)
y_train = pd.read_csv('https://raw.githubusercontent.com/hmkim312/datas/main/HAR/y_train.txt', sep = '\s+',  header = None, names = ['action'])
y_test = pd.read_csv('https://raw.githubusercontent.com/hmkim312/datas/main/HAR/y_test.txt', sep = '\s+',  header = None, names = ['action'])
X_train.shape, X_test.shape, y_train.shape, y_test.shape
X_train.columns = feature_name
X_test.columns = feature_name
X_train.head()
```




<div style="width:100%; height:200px; overflow:auto">
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
      <th>tBodyAcc-mean()-X</th>
      <th>tBodyAcc-mean()-Y</th>
      <th>tBodyAcc-mean()-Z</th>
      <th>tBodyAcc-std()-X</th>
      <th>tBodyAcc-std()-Y</th>
      <th>tBodyAcc-std()-Z</th>
      <th>tBodyAcc-mad()-X</th>
      <th>tBodyAcc-mad()-Y</th>
      <th>tBodyAcc-mad()-Z</th>
      <th>tBodyAcc-max()-X</th>
      <th>...</th>
      <th>fBodyBodyGyroJerkMag-meanFreq()</th>
      <th>fBodyBodyGyroJerkMag-skewness()</th>
      <th>fBodyBodyGyroJerkMag-kurtosis()</th>
      <th>angle(tBodyAccMean,gravity)</th>
      <th>angle(tBodyAccJerkMean),gravityMean)</th>
      <th>angle(tBodyGyroMean,gravityMean)</th>
      <th>angle(tBodyGyroJerkMean,gravityMean)</th>
      <th>angle(X,gravityMean)</th>
      <th>angle(Y,gravityMean)</th>
      <th>angle(Z,gravityMean)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.288585</td>
      <td>-0.020294</td>
      <td>-0.132905</td>
      <td>-0.995279</td>
      <td>-0.983111</td>
      <td>-0.913526</td>
      <td>-0.995112</td>
      <td>-0.983185</td>
      <td>-0.923527</td>
      <td>-0.934724</td>
      <td>...</td>
      <td>-0.074323</td>
      <td>-0.298676</td>
      <td>-0.710304</td>
      <td>-0.112754</td>
      <td>0.030400</td>
      <td>-0.464761</td>
      <td>-0.018446</td>
      <td>-0.841247</td>
      <td>0.179941</td>
      <td>-0.058627</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.278419</td>
      <td>-0.016411</td>
      <td>-0.123520</td>
      <td>-0.998245</td>
      <td>-0.975300</td>
      <td>-0.960322</td>
      <td>-0.998807</td>
      <td>-0.974914</td>
      <td>-0.957686</td>
      <td>-0.943068</td>
      <td>...</td>
      <td>0.158075</td>
      <td>-0.595051</td>
      <td>-0.861499</td>
      <td>0.053477</td>
      <td>-0.007435</td>
      <td>-0.732626</td>
      <td>0.703511</td>
      <td>-0.844788</td>
      <td>0.180289</td>
      <td>-0.054317</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.279653</td>
      <td>-0.019467</td>
      <td>-0.113462</td>
      <td>-0.995380</td>
      <td>-0.967187</td>
      <td>-0.978944</td>
      <td>-0.996520</td>
      <td>-0.963668</td>
      <td>-0.977469</td>
      <td>-0.938692</td>
      <td>...</td>
      <td>0.414503</td>
      <td>-0.390748</td>
      <td>-0.760104</td>
      <td>-0.118559</td>
      <td>0.177899</td>
      <td>0.100699</td>
      <td>0.808529</td>
      <td>-0.848933</td>
      <td>0.180637</td>
      <td>-0.049118</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.279174</td>
      <td>-0.026201</td>
      <td>-0.123283</td>
      <td>-0.996091</td>
      <td>-0.983403</td>
      <td>-0.990675</td>
      <td>-0.997099</td>
      <td>-0.982750</td>
      <td>-0.989302</td>
      <td>-0.938692</td>
      <td>...</td>
      <td>0.404573</td>
      <td>-0.117290</td>
      <td>-0.482845</td>
      <td>-0.036788</td>
      <td>-0.012892</td>
      <td>0.640011</td>
      <td>-0.485366</td>
      <td>-0.848649</td>
      <td>0.181935</td>
      <td>-0.047663</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.276629</td>
      <td>-0.016570</td>
      <td>-0.115362</td>
      <td>-0.998139</td>
      <td>-0.980817</td>
      <td>-0.990482</td>
      <td>-0.998321</td>
      <td>-0.979672</td>
      <td>-0.990441</td>
      <td>-0.942469</td>
      <td>...</td>
      <td>0.087753</td>
      <td>-0.351471</td>
      <td>-0.699205</td>
      <td>0.123320</td>
      <td>0.122542</td>
      <td>0.693578</td>
      <td>-0.615971</td>
      <td>-0.847865</td>
      <td>0.185151</td>
      <td>-0.043892</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 561 columns</p>
</div>



<br>

### 1.3 GBM import 후 실행


```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import time
import warnings

warnings.filterwarnings('ignore')

start_time = time.time()
gb_clf = GradientBoostingClassifier(random_state=13)
gb_clf.fit(X_train, y_train)
gb_pred = gb_clf.predict(X_test)

print('ACC : ', accuracy_score(y_test, gb_pred))
print('Fit time : ', time.time() - start_time)
```

    ACC :  0.9389209365456397
    Fit time :  481.2106680870056


- GBM은 시간이 오래걸림. 확인하기 위해 time으로 확인해봄 (481초 나옴)
- 일반적으로 GBM은 랜덤포레스트보다 좋다고 알려져있음
- ACC는 0.938로 괜찮은편

<br>

### 1.4 GridSearch로 하이퍼파라미터 튜닝을 해보면?


```python
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [100, 500],
    'learning_rate': [0.05, 0.1]
}

start_time = time.time()
grid = GridSearchCV(gb_clf, param_grid=params, cv=2, verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)
print('Fit time : ', time.time()-start_time)
```

    Fitting 2 folds for each of 4 candidates, totalling 8 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 16 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   4 out of   8 | elapsed:  4.5min remaining:  4.5min
    [Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed: 20.4min finished


    Fit time :  3626.6389248371124


- learning_rate : 경사하강법 (Gradient Descent)에서 최저점을 찾아가기 위한 과정으로 너무크면 진동현상이 발생하고, 너무작으면 속도가 느려지며, local mininum에 빠질수 있음
- 그리고 최대 단점인 오래걸림 3600초..

<br>

### 1.5 Best 스코어와 파라미터는?


```python
grid.best_params_
```




    {'learning_rate': 0.1, 'n_estimators': 500}




```python
grid.best_score_
```




    0.9009793253536453



<br>

### 1.6 Test Data의 성능확인


```python
accuracy_score(y_test, grid.best_estimator_.predict(X_test))
```




    0.9419748897183576



- 0.94로 괜찮게 나옴(과적합도 아닌것으로보임)

<br>

## 2. XGBoost
---

### 2.1 XGBoost란?
- 트리 기반의 앙상블 학습에서 가장 각광받는 알고리즘 중에 하나
- GBM 기반의 알고리즘의 느린속도를 다양한 규제를 통해 해결
- 병렬 학습이 가능하도록 설계됨
- XGBoost는 반복 수행시 마다 내부적으로 학습데이터와 검증데이터를 교차검증으로 수행
- 교차검증을 통해 최적화되면 반복을 중단하는 조기 중단 기능이 있음

<br>

### 2.2 설치
- pip install xgboost
- brew install libomp (맥유저)
- XGBoost는 따로 설치를 해야함

<br>

### 2.3 주요 파라미터
- nthread : CPU의 실행 스레드 개수를 조정. 디폴트는 CPU의 전체 스레드를 사용하는것
- eta : GBM 학습률
- num_boost_rounds : n_estimators와 같은 파라미터
- max_depth

<br>

### 2.4 성능 확인


```python
from xgboost import XGBClassifier

start_time = time.time()
xgb = XGBClassifier(n_estimators=400 , learning_rate= 0.1, max_depth=3)
xgb.fit(X_train.values, y_train)
print('Fit time : ', time.time()-start_time)
print('Acc : ',accuracy_score(y_test,xgb.predict(X_test.values)))
```

    Fit time :  40.31653904914856
    Acc :  0.9494401085850017


- fit과 predict 할때 .values를 써야함
- ACC는 0.949가 나옴

<br>

### 2.5 조기종료조건과 검증데이터 지정


```python
from xgboost import XGBClassifier

evals = [(X_test.values, y_test)]

start_time = time.time()
xgb = XGBClassifier(n_estimators = 400, learning_rate = 0.1, max_depth = 3)
xgb.fit(X_train.values, y_train, early_stopping_rounds = 10, eval_set=evals)
print('Fit time : ', time.time() - start_time)
print('Acc : ',accuracy_score(y_test,xgb.predict(X_test.values)))
```

    [0]	validation_0-merror:0.17916
    Will train until validation_0-merror hasn't improved in 10 rounds.
    [1]	validation_0-merror:0.16288
    [2]	validation_0-merror:0.15100
    [3]	validation_0-merror:0.14388
    [4]	validation_0-merror:0.14252
    [5]	validation_0-merror:0.13336
    ...
    [115]	validation_0-merror:0.05972
    [116]	validation_0-merror:0.06006
    [117]	validation_0-merror:0.06006
    [118]	validation_0-merror:0.06006
    [119]	validation_0-merror:0.06006
    Stopping. Best iteration:
    [109]	validation_0-merror:0.05735
    
    Fit time :  17.90309190750122
    Acc :  0.9426535459789617


- early_stopping_rounds 조기 중단을 위한 라운드를 설정, 조기 중단 기능 수행을 위해서는 반드시 eval_set과 eval_metric이 함께 설정되어야 합니다.
    - eval_set : 성능평가를 위한 평가용 데이터 세트를 설정
    - eval_metric : 평가 세트에 적용할 성능 평가 방법
    - (반복마다 eval_set으로 지정된 데이터 세트에서 eval_metric의 지정된 평가 지표로 예측 오류를 측정)
- early_stopping을 하여도 실제 Acc는 큰 차이가 없음 0.949 -> 0.942

<br>

## 3. LightGBM
---

### 3.1 LightGBM
- LightGBM은 XGBoost와 함께 부스팅 계열에서 가장 각광받는 알고리즘
- LGBM의 큰 장점은 속도
- 단, 적은 수의 데이터에는 어울리지 않음 (일반적으로 10000건 이상의 데이터가 필요하다고 함)
- GPU 버전도 존재함

<br>

### 3.2 설치
- brew install lightgbm
- pip install lightgbm

<br>

### 3.3 실행


```python
from lightgbm import LGBMClassifier

start_time = time.time()
lgbm = LGBMClassifier(n_estimator=400)
lgbm.fit(X_train.values, y_train, early_stopping_rounds=100, eval_set=evals)
print('Fit time : ', time.time() - start_time)
print('Acc : ',accuracy_score(y_test, grid.best_estimator_.predict(X_test.values)))
```

    [1]	valid_0's multi_logloss: 1.4404
    Training until validation scores don't improve for 100 rounds
    [2]	valid_0's multi_logloss: 1.21574
    [3]	valid_0's multi_logloss: 1.04795
    [4]	valid_0's multi_logloss: 0.913299
    [5]	valid_0's multi_logloss: 0.812686
    [6]	valid_0's multi_logloss: 0.725964
    [7]	valid_0's multi_logloss: 0.652995
    [8]	valid_0's multi_logloss: 0.591598
    ...
    [95]	valid_0's multi_logloss: 0.266265
    [96]	valid_0's multi_logloss: 0.26572
    [97]	valid_0's multi_logloss: 0.265671
    [98]	valid_0's multi_logloss: 0.265732
    [99]	valid_0's multi_logloss: 0.265704
    [100]	valid_0's multi_logloss: 0.264742
    Did not meet early stopping. Best iteration is:
    [38]	valid_0's multi_logloss: 0.233106
    Fit time :  6.173863887786865
    Acc :  0.9419748897183576


- 속도가 빠름, 6초. 처음 GBM이랑 비교하면 엄청 차이남
- 성능도 0.94로 그전 GBM모델들과 큰 차이가 안남
