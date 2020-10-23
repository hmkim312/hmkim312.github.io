---
title: Mall Customer Segmentation Data로 해보는 군집화
author: HyunMin Kim
date: 2020-10-22 09:10:00 0000
categories: [Data Science, Machine Learning]
tags: [K-Means, Clustering, Kaggle]
---




## 1. Mall Customer Segmentation Data
---
### 1.1 Mall Customer Segmentation Data
- Kaggle에 있는 쇼핑몰 고객 데이터
- <https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python>{:target="_blank"}

<br>

## 2. Mall Customer Segmentation Data 실습
---
### 2.1 Data Load


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('https://raw.githubusercontent.com/hmkim312/datas/main/mallcustomer/Mall_Customers.csv')
dataset.tail()
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
      <th>CustomerID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>195</th>
      <td>196</td>
      <td>Female</td>
      <td>35</td>
      <td>120</td>
      <td>79</td>
    </tr>
    <tr>
      <th>196</th>
      <td>197</td>
      <td>Female</td>
      <td>45</td>
      <td>126</td>
      <td>28</td>
    </tr>
    <tr>
      <th>197</th>
      <td>198</td>
      <td>Male</td>
      <td>32</td>
      <td>126</td>
      <td>74</td>
    </tr>
    <tr>
      <th>198</th>
      <td>199</td>
      <td>Male</td>
      <td>32</td>
      <td>137</td>
      <td>18</td>
    </tr>
    <tr>
      <th>199</th>
      <td>200</td>
      <td>Male</td>
      <td>30</td>
      <td>137</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>



- 총 200개의 데이터 
- id : 고윳값
- gender : 성별, 
- income : 소득, 
- spendig score : 쇼핑몰에서 부여한 고객의 점수 (소비금액 및 행동 패턴 기반)

<br>

### 2.2 5개 군집으로 해보기


```python
from sklearn.cluster import KMeans

X = dataset.iloc[:, [3,4]].values
model = KMeans(n_clusters= 5, init = 'k-means++', random_state = 13)
cluster = model.fit_predict(X)
```

- 데이터에서 income과 Score만 가지고 군집화를 해봄

<br>

### 2.3 시각화


```python
plt.figure(figsize=(12, 10))
plt.scatter(X[cluster == 0, 0], X[cluster == 0, 1], s= 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[cluster == 1, 0], X[cluster == 1, 1], s= 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[cluster == 2, 0], X[cluster == 2, 1], s= 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[cluster == 3, 0], X[cluster == 3, 1], s= 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[cluster == 4, 0], X[cluster == 4, 1], s= 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1 - 100)')
plt.legend()
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/96835668-de1fe880-147e-11eb-9277-b4ae408697fb.png'>


- x = X[cluster == 4, 0], y = X[cluster == 4, 1] 뜻으로, x는 cluster가 4인것의 x축(income), y는 cluster가 4인것의 y축(score)란 뜻
- 같은 의미로 model.cluster.centers_로 각 센터의 x축([:, 0])과 y축([:, 1])임
- 쇼핑몰 고객을 5개의 군집으로 나눈것도 괜찮아 보임
