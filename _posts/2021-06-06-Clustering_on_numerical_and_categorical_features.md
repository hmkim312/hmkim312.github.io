---
title: Numerical and Categorical Features의 Clustering 방법
author: HyunMin Kim
date: 2021-06-06 00:00:00 0000
categories: [Data Science, Machine Learning]
tags: [Gower DIstance, Partial Similarities, Clustering, DBSCAN]
---

## 1. 들어가면서 
---
- Clustering은 비지도 학습에서 많이 쓰는 방법이다. 다만 모든 특성이 Numerical할때 사용하기가 편하다. 왜냐하면 서로의 평균과 거리를 통해서 클러스터링을 하기 때문이다.
- 그래서 Categorical Feature에는 사용하기 어려운게 사실이다.
- Categorical과 Numerical Feature가 있는 Data 일때는 어떤 방식으로 Clustering을 할까? (Clustering : 데이터를 유사한 집단으로 나누는것)
- 해당 내용은 아래의 출처를 보고 작성한 내용입니다.
- 출처 : URL <https://towardsdatascience.com/clustering-on-numerical-and-categorical-features-6e0ebcf1cbad>{:target="_blank"}


## 2. Gower Distance
---
- 만일 결혼이라는 Feature에 미혼, 기혼, 이혼이라는 특성이 있다고 한다면 이는 1[미혼], 2[기혼], 3[이혼]으로 Label Encoding을 할수 있다.
- 하지만 이혼이 기혼보다 싱글이 더 유사하다고는 할수 없습니다.(3[이혼] - 기혼[2] = 1,  3[이혼] - 1[미혼] = 2) 1이 2보다 더 이혼과 미혼, 기혼의 거리를 이야기하기는 이상합니다.
- 여기서 Gower Distance(GD, 1 - Gower Similarity, GS)의 개념이 나옵니다. Gower Similarity는 관측치 i와 j간의 유사성을 계산하기 위해 m개 특성에 대한 부분 유사성의 평균으로 계산됩니다.

<p style="text-align: center;">
    <img src="https://user-images.githubusercontent.com/77366857/120574697-c9ff8900-c45a-11eb-8e58-02809be21cde.png">
</p>

## 3. Partial Similarities
---
- Partial Similarities(PS)는 Numerical과 Categorical에 따라 계산이 다릅니다.
- Numerical은 1에서 두 데이터i와 j 사이의 유사성 부분을 뺀 절대값에 Feature의 범위를 나눠준 값을 뺀것입니다.
<p style="text-align: center;">
    <img src="https://user-images.githubusercontent.com/77366857/120575261-b0ab0c80-c45b-11eb-90b4-c5c0d6f7a0f4.png">
</p>

- Categorical은 모든 특성에 대해 정확히 같은 값을 가질 경우에만 1입니다. 그렇지 않으면 0입니다. 
- Partial Similarities는 항상 0~1사이의 값을 가집니다. 0은 완전히 유사하지 않고 1은 완전히 유사하다는 것을 의미합니다.
- 이러한 결과 떄문에 Categorical Feature와 Numerical Feature가 혼합되어 있을때 Clustering을 할수 있습니다.
    
## 4. Gower Distance의 수학적 특성
---
- Gower Distance(GD)와 Gower Similarity는 유클리드 기하학을 따르지 않는 거리를 사용하기 떄문에 아래의 유클리드 거리를 기반으로하는 Clustering은 사용하지 않습니다.
    - K-means
    - Ward 's, centroid, median method of hierarchical clustering
    
## 5. Gower Distance using Python
---


```python
import pandas as pd

# Creating a dictionary with the data
dictionary = {"age": [22, 25, 30, 38, 42, 47, 55, 62, 61, 90], 
              "gender": ["M", "M", "F", "F", "F", "M", "M", "M", "M", "M"], 
              "civil_status": ["SINGLE", "SINGLE", "SINGLE", "MARRIED", "MARRIED", "SINGLE", "MARRIED", "DIVORCED", "MARRIED", "DIVORCED"], 
              "salary": [18000, 23000, 27000, 32000, 34000, 20000, 40000, 42000, 25000, 70000], 
              "has_children": [False, False, False, True, True, False, False, False, False, True], 
              "purchaser_type": ["LOW_PURCHASER", "LOW_PURCHASER", "LOW_PURCHASER", "HEAVY_PURCHASER", "HEAVY_PURCHASER", "LOW_PURCHASER", "MEDIUM_PURCHASER", "MEDIUM_PURCHASER", "MEDIUM_PURCHASER", "LOW_PURCHASER"]}

# Creating a Pandas DataFrame from the dictionary
dataframe = pd.DataFrame.from_dict(dictionary)
dataframe
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
      <th>age</th>
      <th>gender</th>
      <th>civil_status</th>
      <th>salary</th>
      <th>has_children</th>
      <th>purchaser_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22</td>
      <td>M</td>
      <td>SINGLE</td>
      <td>18000</td>
      <td>False</td>
      <td>LOW_PURCHASER</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25</td>
      <td>M</td>
      <td>SINGLE</td>
      <td>23000</td>
      <td>False</td>
      <td>LOW_PURCHASER</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>F</td>
      <td>SINGLE</td>
      <td>27000</td>
      <td>False</td>
      <td>LOW_PURCHASER</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38</td>
      <td>F</td>
      <td>MARRIED</td>
      <td>32000</td>
      <td>True</td>
      <td>HEAVY_PURCHASER</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>F</td>
      <td>MARRIED</td>
      <td>34000</td>
      <td>True</td>
      <td>HEAVY_PURCHASER</td>
    </tr>
    <tr>
      <th>5</th>
      <td>47</td>
      <td>M</td>
      <td>SINGLE</td>
      <td>20000</td>
      <td>False</td>
      <td>LOW_PURCHASER</td>
    </tr>
    <tr>
      <th>6</th>
      <td>55</td>
      <td>M</td>
      <td>MARRIED</td>
      <td>40000</td>
      <td>False</td>
      <td>MEDIUM_PURCHASER</td>
    </tr>
    <tr>
      <th>7</th>
      <td>62</td>
      <td>M</td>
      <td>DIVORCED</td>
      <td>42000</td>
      <td>False</td>
      <td>MEDIUM_PURCHASER</td>
    </tr>
    <tr>
      <th>8</th>
      <td>61</td>
      <td>M</td>
      <td>MARRIED</td>
      <td>25000</td>
      <td>False</td>
      <td>MEDIUM_PURCHASER</td>
    </tr>
    <tr>
      <th>9</th>
      <td>90</td>
      <td>M</td>
      <td>DIVORCED</td>
      <td>70000</td>
      <td>True</td>
      <td>LOW_PURCHASER</td>
    </tr>
  </tbody>
</table>
</div>



- 임의의 고객 데이터를 만들었습니다. 해당 데이터로 Gower Distance를 계산해보겠습니다.
- 해당 데이터는 총 10명의 고객이 있고 6개의 Feature로 이루어져 있습니다.
    - Age(나이): Numerical
    - Gender(성별): Categorical
    - Civil Status(결혼 상태): Categorical
    - Salary(연봉): Numerical
    - Does the client have children?(아이의 유무): Binary
    - Purchaser Type(구매자 유형): Categorical


```python
data = {'age':[22, 25, '|22-25|/86', 0.44117647],
        'gender' : ['M', 'M', '-', 0],
        'civil_status' : ['SINGLE', 'SINGLE', '-', 0],
        'salary' : [18000, 23000, '|18000-23000|/52000', 0.096153846],
        'has_children' : [False, False, '-', 0],
        'purchaser_type' : ['LOW_PURCHASER', 'LOW_PURCHASER', '-', 0]
       }
pd.DataFrame(data, index = ['Customer_1', 'Customer_2', 'Partial Dissimilarity - Calculation', 'Partial Dissimilarity - Value'])
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
      <th>age</th>
      <th>gender</th>
      <th>civil_status</th>
      <th>salary</th>
      <th>has_children</th>
      <th>purchaser_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Customer_1</th>
      <td>22</td>
      <td>M</td>
      <td>SINGLE</td>
      <td>18000</td>
      <td>False</td>
      <td>LOW_PURCHASER</td>
    </tr>
    <tr>
      <th>Customer_2</th>
      <td>25</td>
      <td>M</td>
      <td>SINGLE</td>
      <td>23000</td>
      <td>False</td>
      <td>LOW_PURCHASER</td>
    </tr>
    <tr>
      <th>Partial Dissimilarity - Calculation</th>
      <td>|22-25|/86</td>
      <td>-</td>
      <td>-</td>
      <td>|18000-23000|/52000</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>Partial Dissimilarity - Value</th>
      <td>0.441176</td>
      <td>0</td>
      <td>0</td>
      <td>0.0961538</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



- Numerical feature는 두 고객 i와 j의 Partial Dissmilarity에서 구한다. salary는 max인 70000과 min인 18000을 뺀 범위로 나누어준다. age도 마찬가지이다.
- Categorical feature는 두 고객 i와 j의 일치한것을 보고 찾는다. 일치하면 0 아니면 1으로 구성된다.
- civil_status, has_children, purchaser_type 모두 일치하기 때문에 0으로 된다.
- Gower Dissimilarity는 모든 Feature의 평균으로 구합니다. 여기서는 (0.441176 + 0 + 0 + 0.0961538 + 0 + 0) / 6 = 0.023379 입니다.


```python
# !pip install gower
import gower

distance_matrix = gower.gower_matrix(dataframe)
columns = [f'Customer_{i}' for i in range(1,11)]
distance_matrix = pd.DataFrame(distance_matrix, index=columns, columns=columns)
```

- 위의 계산 방식을 기반으로 gower를 사용하여 모든 고객들의 GD를 구하였습니다.

## 6. Clustering using DBSCAN
---


```python
from sklearn.cluster import DBSCAN

# Configuring the parameters of the clustering algorithm
dbscan_cluster = DBSCAN(eps=0.3, 
                        min_samples=2, 
                        metric="precomputed")

# Fitting the clustering algorithm
dbscan_cluster.fit(distance_matrix)

# Adding the results to a new column in the dataframe
dataframe["cluster"] = dbscan_cluster.labels_
dataframe
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
      <th>age</th>
      <th>gender</th>
      <th>civil_status</th>
      <th>salary</th>
      <th>has_children</th>
      <th>purchaser_type</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22</td>
      <td>M</td>
      <td>SINGLE</td>
      <td>18000</td>
      <td>False</td>
      <td>LOW_PURCHASER</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25</td>
      <td>M</td>
      <td>SINGLE</td>
      <td>23000</td>
      <td>False</td>
      <td>LOW_PURCHASER</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>F</td>
      <td>SINGLE</td>
      <td>27000</td>
      <td>False</td>
      <td>LOW_PURCHASER</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38</td>
      <td>F</td>
      <td>MARRIED</td>
      <td>32000</td>
      <td>True</td>
      <td>HEAVY_PURCHASER</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>F</td>
      <td>MARRIED</td>
      <td>34000</td>
      <td>True</td>
      <td>HEAVY_PURCHASER</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>47</td>
      <td>M</td>
      <td>SINGLE</td>
      <td>20000</td>
      <td>False</td>
      <td>LOW_PURCHASER</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>55</td>
      <td>M</td>
      <td>MARRIED</td>
      <td>40000</td>
      <td>False</td>
      <td>MEDIUM_PURCHASER</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>62</td>
      <td>M</td>
      <td>DIVORCED</td>
      <td>42000</td>
      <td>False</td>
      <td>MEDIUM_PURCHASER</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>61</td>
      <td>M</td>
      <td>MARRIED</td>
      <td>25000</td>
      <td>False</td>
      <td>MEDIUM_PURCHASER</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>90</td>
      <td>M</td>
      <td>DIVORCED</td>
      <td>70000</td>
      <td>True</td>
      <td>LOW_PURCHASER</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
</div>



- Sklearn의 DBSCAN을 사용하여 Clustering을 하였습니다.
- Gower Distance를 사용하면 K-means의 알고리즘을 사용할수 없기 때문에 DBSCAN을 사용하였습니다.
- DBSCAN이란 밀도 기반의 클러스터링으로, 점이 세밀하게 몰려 있어서 밀도가 높은 부분을 클러스터링 하는 방식으로 어느점을 기준으로 반경 x내에 점이 n개 이상 있으면 하나의 군집으로 인식하는 방식입니다.


- Cluster 0 : 고객들의 연봉이 18,000과 27,000이고 아이가 없으며 자주 구매하지 않는 구매자 유형
- Cluster 1 : 고객들의 연봉이 33,000이고 아이가 있으며 나이는 약 40대의 여성
- Cluster 2 : 60대 남성이며 아이가 없고 자주 구매하는 구매자
- Cluster -1: 어떤 군집에도 포함되지 않는 아웃라이어

## 7. 결론 및 회고
---

- 일단 Clustering은 비지도학습이라 지도학습보다 어려운것이 사실입니다.
- 또한 K-means와 같은 일반적인 Clustering 알고리즘은 대부분 수치형 데이터에만 사용하기가 쉽습니다.
- 하지만 Gower Distance를 Python으로 간단하게 구현할수 있음을 알게 된것을 좋게 생각합니다/
- 나중에 실무에서 Clustering을 하게 된다면 분명 이처럼 Data type이 섞여있을텐데 그때 좋은 예제가 된듯 합니다.
