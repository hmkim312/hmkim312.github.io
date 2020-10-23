---
title: MNIST로 해보는 PCA와 kNN
author: HyunMin Kim
date: 2020-10-23 16:10:00 0000
categories: [Python, Machine Learning]
tags: [kNN, PCA, Mnist, Confusion Matrix]
---

## 1. MNIST
---
### 1.1 MNIST란
- NIST는 미국 국립표준기술연구소(National Institute of Standards and Technology)의 약자입니다. 여기서 진행한 미션 중에 손글씨 데이터를 모았는데, 그중 숫자로 된 데이터를 MNIST라고 합니다.
- 28 * 28 픽셀의 0 ~ 9 사이의 숫자 이미지와 레이블로 구성된 데이터 셋
- 머신러닝 공부하는 사람들이 입문용으로 사용을 많이함
- 60000개의 훈련용 셋과 10000개의 실험용 셋트로 구성되어있음
- 데이터는 kaggle에 있습니다. https://www.kaggle.com/oddrationale/mnist-in-csv

<br>

## 2. PCA와 kNN 실습해보기
---
### 2.1 데이터 로드


```python
import pandas as pd

df_train = pd.read_csv('https://media.githubusercontent.com/media/hmkim312/datas/main/mnist/mnist_train.csv')
df_test = pd.read_csv('https://media.githubusercontent.com/media/hmkim312/datas/main/mnist/mnist_test.csv')

df_train.shape, df_test.shape
```




    ((60000, 785), (10000, 785))



<br>

### 2.2 train 데이터의 모양


```python
df_train.head()
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
      <th>label</th>
      <th>1x1</th>
      <th>1x2</th>
      <th>1x3</th>
      <th>1x4</th>
      <th>1x5</th>
      <th>1x6</th>
      <th>1x7</th>
      <th>1x8</th>
      <th>1x9</th>
      <th>...</th>
      <th>28x19</th>
      <th>28x20</th>
      <th>28x21</th>
      <th>28x22</th>
      <th>28x23</th>
      <th>28x24</th>
      <th>28x25</th>
      <th>28x26</th>
      <th>28x27</th>
      <th>28x28</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 785 columns</p>
</div>

<br>

### 2.3 test 데이터의 모양


```python
df_test.head()
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
      <th>label</th>
      <th>1x1</th>
      <th>1x2</th>
      <th>1x3</th>
      <th>1x4</th>
      <th>1x5</th>
      <th>1x6</th>
      <th>1x7</th>
      <th>1x8</th>
      <th>1x9</th>
      <th>...</th>
      <th>28x19</th>
      <th>28x20</th>
      <th>28x21</th>
      <th>28x22</th>
      <th>28x23</th>
      <th>28x24</th>
      <th>28x25</th>
      <th>28x26</th>
      <th>28x27</th>
      <th>28x28</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 785 columns</p>
</div>



<br>

### 2.4 데이터 정리


```python
import numpy as np

X_train = np.array(df_train.iloc[:, 1:])
y_train = np.array(df_train['label'])

X_test = np.array(df_test.iloc[:,1:])
y_test = np.array(df_test['label'])

X_train.shape, y_train.shape, X_test.shape, y_test.shape
```




    ((60000, 784), (60000,), (10000, 784), (10000,))



- X_train, y_train, X_test, y_test로 데이터를 정리
- label이 y값임

<br>

### 2.5 랜덤하게 16개를 뽑아서 생김새 보기


```python
import random
import matplotlib.pyplot as plt

samples = random.choices(population=range(0, 60000), k=16)

plt.figure(figsize=(14, 12))

for idx, n in enumerate(samples):
    plt.subplot(4, 4, idx + 1)
    plt.imshow(X_train[n].reshape(28, 28),
               cmap='Greys', interpolation='nearest')
    plt.title(y_train[n])

plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/97017808-5246b300-1589-11eb-869d-414ea4c90f53.png'>


- 사람들이 손으로 쓴 글씨로, 헷갈리게 생긴 글씨도 있긴하다

<br>

### 2.6 kNN으로 학습하기


```python
from sklearn.neighbors import KNeighborsClassifier
import time

start_time = time.time()
clf = KNeighborsClassifier(n_neighbors= 5)
clf.fit(X_train, y_train)
print('Fit time : ', time.time() - start_time)
```

    Fit time :  12.099517345428467


- kNN은 따로 학습을 하는 모델은 아니기에 시간이 오래걸리진 않는다

<br>

### 2.7 kNN 성능 확인


```python
from sklearn.metrics import accuracy_score

start_time = time.time()
pred = clf.predict(X_test)
print('Fit time : ', time.time() - start_time)
print(accuracy_score(y_test, pred))
```

    Fit time :  663.8233301639557
    0.9688


- 학습 시간이 너무 오래걸린다

<br>

### 2.8 kNN의 단점 - 차원의 저주
- kNN에서 중요한 것은 차원의 저주(Curse of Dimensionality) 이다. 
- kNN은 차원이 증가하면 feature(머신러닝, 딥러닝에서 input 변수)가 많아지면 학습 데이터의 수가 차원의 수보다 적어져 성능이 저하되는 현상이 생긴다.
- 위의 말은 차원이 증가할수록 개별 차원 내 학습할 데이터 수가 적어지는 현상이 발생하게 되는 이야기이다.
- 이는 공간에 채울 데이터가 많이 필요하고 거리를 계산하는 kNN 알고리즘 특성 상, 거리가 멀어지며 '근접' 이라는 개념이 옅어지게 된다.
- 또한 데이터의 차원이 증가하게 되면 데이터의 거리는 (유클리드거리를 사용한다면) 기하급수적으로 늘어나게 된되며 이는 데이터의 부피가 커진다라는 표현을 사용하기도 한다.
- 차원을 축소시키거나 데이터를 많이 획득하는 방식으로 해결

<br>

### 2.9 차원의 저주를 해결하기 위해 PCA로 차원을 줄여줌


```python
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold

pipe = Pipeline([
    ('pca', PCA()),
    ('clf', KNeighborsClassifier()),
])

parameters = {
    'pca__n_components' : [2, 5, 10],
    'clf__n_neighbors' : [5, 10, 15]
}

kf = StratifiedKFold(n_splits=5, shuffle= True, random_state= 13)
grid = GridSearchCV(pipe, parameters, cv = kf, n_jobs= -1, verbose=1)
grid.fit(X_train, y_train)
```

    Fitting 5 folds for each of 9 candidates, totalling 45 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 16 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  45 out of  45 | elapsed:   34.6s finished





    GridSearchCV(cv=StratifiedKFold(n_splits=5, random_state=13, shuffle=True),
                 estimator=Pipeline(steps=[('pca', PCA()),
                                           ('clf', KNeighborsClassifier())]),
                 n_jobs=-1,
                 param_grid={'clf__n_neighbors': [5, 10, 15],
                             'pca__n_components': [2, 5, 10]},
                 verbose=1)



<br>

### 2.10 best score


```python
print('Best scroe : %0.3f' %grid.best_score_)
print('Best parameters set:')
best_parameters = grid.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r'%(param_name, best_parameters[param_name]))
```

    Best scroe : 0.931
    Best parameters set:
    	clf__n_neighbors: 10
    	pca__n_components: 10


<br>

### 2.11 pca를 하여 차원 축소를 한 Accuracy


```python
accuracy_score(y_test, grid.best_estimator_.predict(X_test))
```




    0.9286



- 약 0.928의 Accuracy를 가진다. 
- 학습 시간도 얼마 걸리지 않았음

<br>

### 2.12 Confusion_matrix로 결과 확인


```python
def results(y_pred, y_test):
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(y_test, y_pred))
    
results(grid.predict(X_train), y_train)
```

                  precision    recall  f1-score   support
    
               0       0.96      0.98      0.97      5923
               1       0.98      0.99      0.98      6742
               2       0.96      0.96      0.96      5958
               3       0.94      0.90      0.92      6131
               4       0.94      0.93      0.93      5842
               5       0.93      0.94      0.93      5421
               6       0.96      0.98      0.97      5918
               7       0.96      0.95      0.96      6265
               8       0.92      0.91      0.91      5851
               9       0.90      0.91      0.90      5949
    
        accuracy                           0.94     60000
       macro avg       0.94      0.94      0.94     60000
    weighted avg       0.94      0.94      0.94     60000
    


- 골고루 잘 맞추고 있는것을 확인 가능

<br>

### 2.13 숫자 확인


```python
n = 700
plt.grid(False)
plt.imshow(X_test[n].reshape(28,28), cmap ='Greys',interpolation='nearest')
plt.show()

print('Answer is: ', grid.best_estimator_.predict(X_test[n].reshape(1,784)))
print('Real Label is :', y_test[n])
```

<img src = 'https://user-images.githubusercontent.com/60168331/97017816-54107680-1589-11eb-97a4-1b765f21dd1e.png'>


    Answer is:  [1]
    Real Label is : 1


<br>

### 2.14 틀린 데이터 확인을 위한 전처리


```python
preds = grid.best_estimator_.predict(X_test)
preds
```




    array([7, 2, 1, ..., 4, 5, 6])




```python
y_test
```




    array([7, 2, 1, ..., 4, 5, 6])



<br>

### 2.15 틀린 데이터 그림으로 확인


```python
wrong_results = X_test[y_test != preds]
samples = random.choices(population=range(0, wrong_results.shape[0]), k=16)

plt.figure(figsize=(14, 12))

for idx, n in enumerate(samples):
    plt.grid(False)
    plt.subplot(4, 4, idx + 1)
    plt.imshow(wrong_results[n].reshape(28, 28),
               cmap='Greys', interpolation='nearest')
    plt.title(grid.best_estimator_.predict(wrong_results[n].reshape(1,784))[0])
    
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/97017817-54a90d00-1589-11eb-9778-821cf37cc759.png'>


- 그림의 타이틀은 잘못 예측한 값이다.
- 헷갈릴만한 숫자들이 생각보다 있다.
