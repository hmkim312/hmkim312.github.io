---
title: 하이퍼파라미터 튜닝(Hyperparameter Tuning)
author: HyunMin Kim
date: 2020-09-21 15:00:00 0000
categories: [Data Science, Machine Learning]
tags: [Hyperparameter Tuning, Grid Search]
---

## 1. 하이퍼 파라미터 튜닝
- 모델의 성능을 확보하기 위해 조절하는 설정값

### 1.1 튜닝대상
- 결정나무에서 아직 우리가 튜닝해볼만한 것은 max_depth이다.
- 간단하게 반복문으로 max_depth를 바꿔가며 테스트해볼 수 있을 것이다
- 그런데 앞으로를 생각해서 보다 간편하고 유용한 방법을 생각해보자

### 1.2 데이터 불러오기

```python
import pandas as pd

red_wine = pd.read_csv(red_url, sep = ';')
white_wine = pd.read_csv(white_url, sep = ';')

red_wine['color'] = 1
white_wine['color'] = 0

wine = pd.concat([red_wine, white_wine])
wine['taste'] = [1. if grade > 5 else 0. for grade in wine['quality']]

X = wine.drop(['taste', 'quality'], axis = 1)
y = wine['taste']
```

### 1.3 GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

params = {'max_depth': [2, 4, 7, 10]}
wine_tree = DecisionTreeClassifier(max_depth=2, random_state=13)

gridsearch = GridSearchCV(
    estimator=wine_tree, param_grid=params, cv=5, n_jobs=-1)
gridsearch.fit(X, y)
```




    GridSearchCV(cv=5,
                 estimator=DecisionTreeClassifier(max_depth=2, random_state=13),
                 n_jobs=-1, param_grid={'max_depth': [2, 4, 7, 10]})

- 결과를 확인하고 싶은 파라미터를 Gridsearch를 통해서 정의
- cv는 cross validation
- 여기서 n_jobs 옵션을 높여주면 CPU의 코어를 보다 병렬로 활용함. Core가 많으면 n_jobs를 높이면 속도가 빨라짐

### 1.4 결과


```python
import pprint
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(gridsearch.cv_results_)
```

    {   'mean_fit_time': array([0.01350098, 0.01816139, 0.02338843, 0.03536773]),
        'mean_score_time': array([0.00353079, 0.00311818, 0.00243087, 0.00291348]),
        'mean_test_score': array([0.6888005 , 0.66356523, 0.65340854, 0.64401587]),
        'param_max_depth': masked_array(data=[2, 4, 7, 10],
                 mask=[False, False, False, False],
           fill_value='?',
                dtype=object),
        'params': [   {'max_depth': 2},
                      {'max_depth': 4},
                      {'max_depth': 7},
                      {'max_depth': 10}],
        'rank_test_score': array([1, 2, 3, 4], dtype=int32),
        'split0_test_score': array([0.55230769, 0.51230769, 0.50846154, 0.51615385]),
        'split1_test_score': array([0.68846154, 0.63153846, 0.60307692, 0.60076923]),
        'split2_test_score': array([0.71439569, 0.72363356, 0.68360277, 0.66743649]),
        'split3_test_score': array([0.73210162, 0.73210162, 0.73672055, 0.71054657]),
        'split4_test_score': array([0.75673595, 0.7182448 , 0.73518091, 0.72517321]),
        'std_fit_time': array([0.00195601, 0.00162868, 0.00105901, 0.00375986]),
        'std_score_time': array([0.00033869, 0.00025196, 0.00011653, 0.00041979]),
        'std_test_score': array([0.07179934, 0.08390453, 0.08727223, 0.07717557])}


### 1.5 최적의 성능을 가진 모델은?


```python
gridsearch.best_estimator_.get_params()
```




    {'ccp_alpha': 0.0,
     'class_weight': None,
     'criterion': 'gini',
     'max_depth': 2,
     'max_features': None,
     'max_leaf_nodes': None,
     'min_impurity_decrease': 0.0,
     'min_impurity_split': None,
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'min_weight_fraction_leaf': 0.0,
     'presort': 'deprecated',
     'random_state': 13,
     'splitter': 'best'}

- 그리스 서치를 사용하여 가장 좋은 성능을 가진 파라미터를 출력


```python
gridsearch.best_score_
```




    0.6888004974240539

- 그리드 서치를 사용하여 가장 좋은 스코어를 출력 


```python
gridsearch.best_params_
```




    {'max_depth': 2}

- 그리스 서치를 사용하여 가장 좋은 스코어를 낸 파라미터를 출력
- 그렇다면 그리드 서치와 파이프라인을 같이 할순 없을까

### 1.6 pipeline을 적용한 모델에 gridsearch를 적용

```python
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

estimators = [('scaler', StandardScaler()),
               ('clf', DecisionTreeClassifier(random_state=13))]

pipe = Pipeline(estimators)
```

### 1.7 위에 만든 pipeline을 gridsearch에 적용


```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

params_grid = [{'clf__max_depth': [2, 4, 7, 10]}]

gridsearch = GridSearchCV(
    estimator=pipe, param_grid=params_grid, cv=5, n_jobs=-1)
gridsearch.fit(X, y)
```




    GridSearchCV(cv=5,
                 estimator=Pipeline(steps=[('scaler', StandardScaler()),
                                           ('clf',
                                            DecisionTreeClassifier(random_state=13))]),
                 n_jobs=-1, param_grid=[{'clf__max_depth': [2, 4, 7, 10]}])


- 그리스 서치의 estimator에 pipeline을 적용시켜 주면됨

### 1.8 best model은?


```python
gridsearch.best_estimator_.get_params()
```




    {'memory': None,
     'steps': [('scaler', StandardScaler()),
      ('clf', DecisionTreeClassifier(max_depth=2, random_state=13))],
     'verbose': False,
     'scaler': StandardScaler(),
     'clf': DecisionTreeClassifier(max_depth=2, random_state=13),
     'scaler__copy': True,
     'scaler__with_mean': True,
     'scaler__with_std': True,
     'clf__ccp_alpha': 0.0,
     'clf__class_weight': None,
     'clf__criterion': 'gini',
     'clf__max_depth': 2,
     'clf__max_features': None,
     'clf__max_leaf_nodes': None,
     'clf__min_impurity_decrease': 0.0,
     'clf__min_impurity_split': None,
     'clf__min_samples_leaf': 1,
     'clf__min_samples_split': 2,
     'clf__min_weight_fraction_leaf': 0.0,
     'clf__presort': 'deprecated',
     'clf__random_state': 13,
     'clf__splitter': 'best'}

- 베스트 모델의 파라미터 확인
- gridsearch.best_estimator_ 가장 성능이 좋은 모델을 저장시켜놓은것
- 위의 모델로 predict도 가능함.

### 1.9 best_score_


```python
gridsearch.best_score_
```




    0.6888004974240539




```python
# split0_test_score 에서 2, 4, 7, 10 순으로 depth를 적용시켜 한것 각 array에 0번이 depth2 이런식임
gridsearch.cv_results_
```




    {'mean_fit_time': array([0.01857271, 0.02339497, 0.0245254 , 0.0390317 ]),
     'std_fit_time': array([0.00268332, 0.00160271, 0.00083077, 0.00548869]),
     'mean_score_time': array([0.00390778, 0.00349479, 0.00230117, 0.00337038]),
     'std_score_time': array([3.74068542e-04, 2.00201510e-04, 5.41403186e-05, 5.70083055e-04]),
     'param_clf__max_depth': masked_array(data=[2, 4, 7, 10],
                  mask=[False, False, False, False],
            fill_value='?',
                 dtype=object),
     'params': [{'clf__max_depth': 2},
      {'clf__max_depth': 4},
      {'clf__max_depth': 7},
      {'clf__max_depth': 10}],
     'split0_test_score': array([0.55230769, 0.51230769, 0.50846154, 0.51615385]),
     'split1_test_score': array([0.68846154, 0.63153846, 0.60461538, 0.60230769]),
     'split2_test_score': array([0.71439569, 0.72363356, 0.68206313, 0.66589684]),
     'split3_test_score': array([0.73210162, 0.73210162, 0.73672055, 0.71054657]),
     'split4_test_score': array([0.75673595, 0.7182448 , 0.73518091, 0.72517321]),
     'mean_test_score': array([0.6888005 , 0.66356523, 0.6534083 , 0.64401563]),
     'std_test_score': array([0.07179934, 0.08390453, 0.08699322, 0.0769154 ]),
     'rank_test_score': array([1, 2, 3, 4], dtype=int32)}



### 1.10 결과를 표로 정리하기


```python
import pandas as pd

score_df = pd.DataFrame(gridsearch.cv_results_)
score_df[['params', 'rank_test_score', 'mean_test_score', 'std_test_score']]
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
      <th>params</th>
      <th>rank_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'clf__max_depth': 2}</td>
      <td>1</td>
      <td>0.688800</td>
      <td>0.071799</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{'clf__max_depth': 4}</td>
      <td>2</td>
      <td>0.663565</td>
      <td>0.083905</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{'clf__max_depth': 7}</td>
      <td>3</td>
      <td>0.653408</td>
      <td>0.086993</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{'clf__max_depth': 10}</td>
      <td>4</td>
      <td>0.644016</td>
      <td>0.076915</td>
    </tr>
  </tbody>
</table>
</div>

- accuracy의 평균과 표준편차를 확인
- score_df의 컬럼을 보면 다른 수치들(fit time 등도 있음)
