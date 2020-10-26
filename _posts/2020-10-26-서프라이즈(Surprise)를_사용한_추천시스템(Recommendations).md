---
title: 서프라이즈(Surprise)를 사용한 추천시스템(Recommendations)
author: HyunMin Kim
date: 2020-10-26 09:00:00 0000
categories: [Data Science, Machine Learning]
tags: [Recommendations, Surprise, Collaborative Filtering]
---


## 1. Surprise
---
### 1.1 Surprise란?

<https://antilibrary.org/2086>{:target="_blank"}

- Surprise 는 추천 시스템을위한 사용하기 쉬운 Python scikit
- pip install scikit-surprise

<br>

### 1.2 Surprise 장점

- 다양한 추천 알고리즘 탑재
- Surprise는 sklearn과 API의 명칭과 속성이 아주 유사함

<br>

## 2. 무비렌즈 라지데이터 실습
---
### 2.1 모듈 Import


```python
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
```

- 보다시피 사이킷런과 비슷한 API를 가지고 있음

<br>

### 2.2 무비렌즈 데이터


```python
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.25, random_state=0)
```

- y를 입력하고 엔터를 누르거나, 에러가 뜨면 홈디렉토리에 .surprise_data 를 만들어서 데이터를 넣으면됨
- 데이터 다운 : <http://files.grouplens.org/datasets/movielens/ml-100k.zip>{:target="_blank}

<br>

### 2.3 잠재요인 협업필터링


```python
algo = SVD()
algo.fit(trainset)
```




    <surprise.prediction_algorithms.matrix_factorization.SVD at 0x7fa2910b2c70>



- SVD 모듈을 사용하여 잠재요인 협업필터링을 사용함

<br>

### 2.4 Test


```python
predictions = algo.test(testset)
print('prediction type : ', type(predictions), 'size : ', len(predictions))
print('prediction 결과의 최초 5개 추출')
predictions[:5]
```

    prediction type :  <class 'list'> size :  25000
    prediction 결과의 최초 5개 추출

    [Prediction(uid='120', iid='282', r_ui=4.0, est=3.7907894998845144, details={'was_impossible': False}),
     Prediction(uid='882', iid='291', r_ui=4.0, est=3.890238997290443, details={'was_impossible': False}),
     Prediction(uid='535', iid='507', r_ui=5.0, est=4.062451393436086, details={'was_impossible': False}),
     Prediction(uid='697', iid='244', r_ui=5.0, est=3.7937949762331944, details={'was_impossible': False}),
     Prediction(uid='751', iid='385', r_ui=4.0, est=3.4501246214582117, details={'was_impossible': False})]



- Test 메서드는 사용자-아이템 평점 데이터 세트 전체에 대해서 추천을 수행함

<br>

### 2.5 uid, iid, est 추출


```python
[(pred.uid, pred.iid, pred.est) for pred in predictions[:3]]
```

    [('120', '282', 3.7907894998845144),
     ('882', '291', 3.890238997290443),
     ('535', '507', 4.062451393436086)]



- uid : 유저 아이디
- iid : 아이템 아이디(영화 아이디)
- r_ui : 실제 평점
- est : 예측 평점

<br>

### 2.6 Predict


```python
uid = str(196)
iid = str(302)
pred = algo.predict(uid, iid)
print(pred)
```

    user: 196        item: 302        r_ui = None   est = 4.11   {'was_impossible': False}


- Predict 메서드는 개별 사용자와 개별 영화에 대한 추천 평점을 반환함
- 주의해야할것은 uid와 iid를 문자열로 입력해야함

<br>

### 2.7 평가 결과


```python
accuracy.rmse(predictions)
```

    RMSE: 0.9503
    0.9502777153756664



- RMSE는 0.95가 나온다

<br>

## 3. 무비 렌즈 스몰데이터 
---
### 3.1 csv 파일 접근


```python
import pandas as pd

ratings = pd.read_csv('https://media.githubusercontent.com/media/hmkim312/datas/main/ml-latest-small/ratings.csv')
ratings.to_csv('./data/ml-latest-small/ratings_noh.csv', index = False, header = False)
```

- surprise는 파일을 읽을때 컬럼명(헤더)가 있으면 안됨
- 그래서 헤더와 인덱스가 없는 csv파일을 ratings_noh.csv로 생성한것
- 순서는 usrID, movieID, rating, timestamp임

<br>

### 3.2 surprise에서 읽기


```python
from surprise import Reader

reader = Reader(line_format='user item rating timestamp', sep = ',', rating_scale=(0.5, 5))
data = Dataset.load_from_file('./data/ml-latest-small/ratings_noh.csv', reader=reader)
```

- rating_scale : 평점값의 최소, 최대 값을 설정

<br>

### 3.3 학습


```python
trainset, testset = train_test_split(data, test_size=0.25, random_state=0)
algo = SVD(n_factors=50, random_state=13)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)
```

    RMSE: 0.8671
    0.8671265529441645



- n_factor : 잠재요인의 크기

<br>

### 3.4 Pandas의 데이터를 바로 읽을수 있음


```python
import pandas as pd
from surprise import Reader, Dataset

ratings = pd.read_csv('./data/ml-latest-small/ratings.csv')
reader = Reader(rating_scale=(0.5, 5.0))

# ratings DataFrame에서 컬럼은 사용자 아이디, 아이템 아이디, 평점 순서를 지켜야함
data = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)
trainset, testset = train_test_split(data, test_size=0.25, random_state=0)

algo = SVD(n_factors=50, random_state=13)
algo.fit(trainset)
predictions = algo.test(testset)
```

- 단 user item rating timestamp 순서로 있어야 함

<br>

### 3.5 CrossValidation 지원


```python
from surprise.model_selection import cross_validate

ratings = pd.read_csv('https://media.githubusercontent.com/media/hmkim312/datas/main/ml-latest-small/ratings.csv')
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)

algo = SVD(random_state=13)
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv = 5, verbose = True)
```

    Evaluating RMSE, MAE of algorithm SVD on 5 split(s).
    
                      Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
    RMSE (testset)    0.8726  0.8780  0.8750  0.8722  0.8666  0.8729  0.0038  
    MAE (testset)     0.6717  0.6723  0.6694  0.6689  0.6688  0.6702  0.0015  
    Fit time          3.32    3.55    3.52    3.55    3.43    3.47    0.09    
    Test time         0.08    0.12    0.21    0.09    0.10    0.12    0.04    

    {'test_rmse': array([0.87261874, 0.87804878, 0.87500197, 0.87224824, 0.86662658]),
     'test_mae': array([0.67167342, 0.67227455, 0.66939226, 0.66888584, 0.66878608]),
     'fit_time': (3.323078155517578,
      3.5504040718078613,
      3.5200130939483643,
      3.5460398197174072,
      3.4292070865631104),
     'test_time': (0.08444976806640625,
      0.11795592308044434,
      0.20586800575256348,
      0.09193277359008789,
      0.09651422500610352)}



<br>

### 3.6 GridSearchCV도 지원


```python
from surprise.model_selection import GridSearchCV

param_grid = {'n_epochs': [20, 40, 60], 'n_factors': [50, 100, 200]}
gs = GridSearchCV(SVD, param_grid, measures=['RMSE', 'MAE'], cv = 3)
gs.fit(data)

print(gs.best_score['rmse'])
print(gs.best_params['rmse'])
```

    0.8770725465323782
    {'n_epochs': 20, 'n_factors': 50}

