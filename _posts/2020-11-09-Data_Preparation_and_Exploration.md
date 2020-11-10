---
title: Porto Seguro’s Safe Driver Prediction 데이터 Preparation & Exploration
author: HyunMin Kim
date: 2020-11-09 00:30:00 0000
categories: [Kaggle, Porto Seguro’s Safe Driver Prediction]
tags: [Kaggle Transcription, Porto Seguro’s Safe Driver Prediction, EDA, Sklearn, Random Forest]
---


## 1. Porto Seguro Safe Driver Prediction
---
### 1.1 Porto Seguro Safe Driver Prediction
- Porto Seguro는 브라질의 자동차 보험회사로, 어떤 차주가 내년에 보험을 청구할지에 대한 예측을 하는것
- <https://www.kaggle.com/c/porto-seguro-safe-driver-prediction>{:target="_blank"}

<br>

## 2. Introduction
---
### 2.1 Introduction
- 이 노트북은 PorteSeguro 대회의 데이터에서 좋은 통찰력을 얻는 것을 목표로합니다. 그 외에도 모델링을 위해 데이터를 준비하는 몇 가지 팁과 요령을 제공합니다. 노트북은 다음과 같은 주요 섹션으로 구성됩니다.

<br>

### 2.2 Sections

- Visual inspection of your data
- Defining the metadata
- Descriptive statistics
- Handling imbalanced classes
- Data quality checks
- Exploratory data visualization
- Feature engineering
- Feature selection
- Feature scaling
- Loading packages

<br>

### 2.3 출처

- <https://www.kaggle.com/bertcarremans/data-preparation-exploration>{:target="_blank"}

<br>

## 3. Loading packages
---
### 3.1 패키지 로드


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', 100)
```

<br>

## 4. Visual inspection of your data
---
### 4.1 Data Load


```python
train = pd.read_csv('https://media.githubusercontent.com/media/hmkim312/datas/main/porto-seguro-safe-driver-prediction/train.csv')
test = pd.read_csv('https://media.githubusercontent.com/media/hmkim312/datas/main/porto-seguro-safe-driver-prediction/test.csv')
```

<br>

### 4.2 Data at first sight


```python
train.head()
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
      <th>id</th>
      <th>target</th>
      <th>ps_ind_01</th>
      <th>ps_ind_02_cat</th>
      <th>ps_ind_03</th>
      <th>ps_ind_04_cat</th>
      <th>ps_ind_05_cat</th>
      <th>ps_ind_06_bin</th>
      <th>ps_ind_07_bin</th>
      <th>ps_ind_08_bin</th>
      <th>ps_ind_09_bin</th>
      <th>ps_ind_10_bin</th>
      <th>ps_ind_11_bin</th>
      <th>ps_ind_12_bin</th>
      <th>ps_ind_13_bin</th>
      <th>ps_ind_14</th>
      <th>ps_ind_15</th>
      <th>ps_ind_16_bin</th>
      <th>ps_ind_17_bin</th>
      <th>ps_ind_18_bin</th>
      <th>ps_reg_01</th>
      <th>ps_reg_02</th>
      <th>ps_reg_03</th>
      <th>ps_car_01_cat</th>
      <th>ps_car_02_cat</th>
      <th>ps_car_03_cat</th>
      <th>ps_car_04_cat</th>
      <th>ps_car_05_cat</th>
      <th>ps_car_06_cat</th>
      <th>ps_car_07_cat</th>
      <th>ps_car_08_cat</th>
      <th>ps_car_09_cat</th>
      <th>ps_car_10_cat</th>
      <th>ps_car_11_cat</th>
      <th>ps_car_11</th>
      <th>ps_car_12</th>
      <th>ps_car_13</th>
      <th>ps_car_14</th>
      <th>ps_car_15</th>
      <th>ps_calc_01</th>
      <th>ps_calc_02</th>
      <th>ps_calc_03</th>
      <th>ps_calc_04</th>
      <th>ps_calc_05</th>
      <th>ps_calc_06</th>
      <th>ps_calc_07</th>
      <th>ps_calc_08</th>
      <th>ps_calc_09</th>
      <th>ps_calc_10</th>
      <th>ps_calc_11</th>
      <th>ps_calc_12</th>
      <th>ps_calc_13</th>
      <th>ps_calc_14</th>
      <th>ps_calc_15_bin</th>
      <th>ps_calc_16_bin</th>
      <th>ps_calc_17_bin</th>
      <th>ps_calc_18_bin</th>
      <th>ps_calc_19_bin</th>
      <th>ps_calc_20_bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.7</td>
      <td>0.2</td>
      <td>0.718070</td>
      <td>10</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>2</td>
      <td>0.400000</td>
      <td>0.883679</td>
      <td>0.370810</td>
      <td>3.605551</td>
      <td>0.6</td>
      <td>0.5</td>
      <td>0.2</td>
      <td>3</td>
      <td>1</td>
      <td>10</td>
      <td>1</td>
      <td>10</td>
      <td>1</td>
      <td>5</td>
      <td>9</td>
      <td>1</td>
      <td>5</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.8</td>
      <td>0.4</td>
      <td>0.766078</td>
      <td>11</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>19</td>
      <td>3</td>
      <td>0.316228</td>
      <td>0.618817</td>
      <td>0.388716</td>
      <td>2.449490</td>
      <td>0.3</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>5</td>
      <td>8</td>
      <td>1</td>
      <td>7</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13</td>
      <td>0</td>
      <td>5</td>
      <td>4</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.000000</td>
      <td>7</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>14</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>60</td>
      <td>1</td>
      <td>0.316228</td>
      <td>0.641586</td>
      <td>0.347275</td>
      <td>3.316625</td>
      <td>0.5</td>
      <td>0.7</td>
      <td>0.1</td>
      <td>2</td>
      <td>2</td>
      <td>9</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>7</td>
      <td>4</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.9</td>
      <td>0.2</td>
      <td>0.580948</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>104</td>
      <td>1</td>
      <td>0.374166</td>
      <td>0.542949</td>
      <td>0.294958</td>
      <td>2.000000</td>
      <td>0.6</td>
      <td>0.9</td>
      <td>0.1</td>
      <td>2</td>
      <td>4</td>
      <td>7</td>
      <td>1</td>
      <td>8</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.7</td>
      <td>0.6</td>
      <td>0.840759</td>
      <td>11</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>14</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>82</td>
      <td>3</td>
      <td>0.316070</td>
      <td>0.565832</td>
      <td>0.365103</td>
      <td>2.000000</td>
      <td>0.4</td>
      <td>0.6</td>
      <td>0.0</td>
      <td>2</td>
      <td>2</td>
      <td>6</td>
      <td>3</td>
      <td>10</td>
      <td>2</td>
      <td>12</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.tail()
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
      <th>id</th>
      <th>target</th>
      <th>ps_ind_01</th>
      <th>ps_ind_02_cat</th>
      <th>ps_ind_03</th>
      <th>ps_ind_04_cat</th>
      <th>ps_ind_05_cat</th>
      <th>ps_ind_06_bin</th>
      <th>ps_ind_07_bin</th>
      <th>ps_ind_08_bin</th>
      <th>ps_ind_09_bin</th>
      <th>ps_ind_10_bin</th>
      <th>ps_ind_11_bin</th>
      <th>ps_ind_12_bin</th>
      <th>ps_ind_13_bin</th>
      <th>ps_ind_14</th>
      <th>ps_ind_15</th>
      <th>ps_ind_16_bin</th>
      <th>ps_ind_17_bin</th>
      <th>ps_ind_18_bin</th>
      <th>ps_reg_01</th>
      <th>ps_reg_02</th>
      <th>ps_reg_03</th>
      <th>ps_car_01_cat</th>
      <th>ps_car_02_cat</th>
      <th>ps_car_03_cat</th>
      <th>ps_car_04_cat</th>
      <th>ps_car_05_cat</th>
      <th>ps_car_06_cat</th>
      <th>ps_car_07_cat</th>
      <th>ps_car_08_cat</th>
      <th>ps_car_09_cat</th>
      <th>ps_car_10_cat</th>
      <th>ps_car_11_cat</th>
      <th>ps_car_11</th>
      <th>ps_car_12</th>
      <th>ps_car_13</th>
      <th>ps_car_14</th>
      <th>ps_car_15</th>
      <th>ps_calc_01</th>
      <th>ps_calc_02</th>
      <th>ps_calc_03</th>
      <th>ps_calc_04</th>
      <th>ps_calc_05</th>
      <th>ps_calc_06</th>
      <th>ps_calc_07</th>
      <th>ps_calc_08</th>
      <th>ps_calc_09</th>
      <th>ps_calc_10</th>
      <th>ps_calc_11</th>
      <th>ps_calc_12</th>
      <th>ps_calc_13</th>
      <th>ps_calc_14</th>
      <th>ps_calc_15_bin</th>
      <th>ps_calc_16_bin</th>
      <th>ps_calc_17_bin</th>
      <th>ps_calc_18_bin</th>
      <th>ps_calc_19_bin</th>
      <th>ps_calc_20_bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>595207</th>
      <td>1488013</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.5</td>
      <td>0.3</td>
      <td>0.692820</td>
      <td>10</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>31</td>
      <td>3</td>
      <td>0.374166</td>
      <td>0.684631</td>
      <td>0.385487</td>
      <td>2.645751</td>
      <td>0.4</td>
      <td>0.5</td>
      <td>0.3</td>
      <td>3</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>12</td>
      <td>4</td>
      <td>1</td>
      <td>9</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>595208</th>
      <td>1488016</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.9</td>
      <td>0.7</td>
      <td>1.382027</td>
      <td>9</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>63</td>
      <td>2</td>
      <td>0.387298</td>
      <td>0.972145</td>
      <td>-1.000000</td>
      <td>3.605551</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>2</td>
      <td>4</td>
      <td>8</td>
      <td>6</td>
      <td>8</td>
      <td>2</td>
      <td>12</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>595209</th>
      <td>1488017</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.9</td>
      <td>0.2</td>
      <td>0.659071</td>
      <td>7</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>31</td>
      <td>3</td>
      <td>0.397492</td>
      <td>0.596373</td>
      <td>0.398748</td>
      <td>1.732051</td>
      <td>0.4</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>3</td>
      <td>2</td>
      <td>7</td>
      <td>4</td>
      <td>8</td>
      <td>0</td>
      <td>10</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>595210</th>
      <td>1488021</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.9</td>
      <td>0.4</td>
      <td>0.698212</td>
      <td>11</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>101</td>
      <td>3</td>
      <td>0.374166</td>
      <td>0.764434</td>
      <td>0.384968</td>
      <td>3.162278</td>
      <td>0.0</td>
      <td>0.7</td>
      <td>0.0</td>
      <td>4</td>
      <td>0</td>
      <td>9</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
      <td>11</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>595211</th>
      <td>1488027</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.1</td>
      <td>0.2</td>
      <td>-1.000000</td>
      <td>7</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>34</td>
      <td>2</td>
      <td>0.400000</td>
      <td>0.932649</td>
      <td>0.378021</td>
      <td>3.741657</td>
      <td>0.4</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>2</td>
      <td>3</td>
      <td>10</td>
      <td>4</td>
      <td>10</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



- 유사한 그룹에 속하는 기능은 기능 이름 (예 : ind, reg, car, calc)에 태그가 지정됩니다.
- 기능 이름에는 이진 기능을 나타내는 접미사 bin과 범주 기능을 나타내는 cat이 포함됩니다.
- 이러한 지정이없는 특징은 연속 형이거나 순서 형입니다.
- -1 값은 관측치에서 피쳐가 누락되었음을 나타냅니다.
- target 열은 해당 보험 계약자에 대한 청구가 접수되었는지 여부를 나타냅니다.

<br>


```python
train.shape
```




    (595212, 59)



- Train Data는 59개의 Column과 595,212개의 Row로 이루어져있습니다.

<br>


```python
train.drop_duplicates()
train.shape
```




    (595212, 59)



- 중복된 데이터가 있는지 확인하기 위해 drop_duplicates를 해보았고, 중복된 데이터는 없는것을 확인하였습니다.

<br>


```python
test.shape
```




    (892816, 58)



- Test data는 Train data와 비교하여 1개의 column이 부족하지만, 이것은 target column입니다.

<br>


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 595212 entries, 0 to 595211
    Data columns (total 59 columns):
     #   Column          Non-Null Count   Dtype  
    ---  ------          --------------   -----  
     0   id              595212 non-null  int64  
     1   target          595212 non-null  int64  
     2   ps_ind_01       595212 non-null  int64  
     3   ps_ind_02_cat   595212 non-null  int64  
     4   ps_ind_03       595212 non-null  int64  
     5   ps_ind_04_cat   595212 non-null  int64  
     6   ps_ind_05_cat   595212 non-null  int64  
     7   ps_ind_06_bin   595212 non-null  int64  
     8   ps_ind_07_bin   595212 non-null  int64  
     9   ps_ind_08_bin   595212 non-null  int64  
     10  ps_ind_09_bin   595212 non-null  int64  
     11  ps_ind_10_bin   595212 non-null  int64  
     12  ps_ind_11_bin   595212 non-null  int64  
     13  ps_ind_12_bin   595212 non-null  int64  
     14  ps_ind_13_bin   595212 non-null  int64  
     15  ps_ind_14       595212 non-null  int64  
     16  ps_ind_15       595212 non-null  int64  
     17  ps_ind_16_bin   595212 non-null  int64  
     18  ps_ind_17_bin   595212 non-null  int64  
     19  ps_ind_18_bin   595212 non-null  int64  
     20  ps_reg_01       595212 non-null  float64
     21  ps_reg_02       595212 non-null  float64
     22  ps_reg_03       595212 non-null  float64
     23  ps_car_01_cat   595212 non-null  int64  
     24  ps_car_02_cat   595212 non-null  int64  
     25  ps_car_03_cat   595212 non-null  int64  
     26  ps_car_04_cat   595212 non-null  int64  
     27  ps_car_05_cat   595212 non-null  int64  
     28  ps_car_06_cat   595212 non-null  int64  
     29  ps_car_07_cat   595212 non-null  int64  
     30  ps_car_08_cat   595212 non-null  int64  
     31  ps_car_09_cat   595212 non-null  int64  
     32  ps_car_10_cat   595212 non-null  int64  
     33  ps_car_11_cat   595212 non-null  int64  
     34  ps_car_11       595212 non-null  int64  
     35  ps_car_12       595212 non-null  float64
     36  ps_car_13       595212 non-null  float64
     37  ps_car_14       595212 non-null  float64
     38  ps_car_15       595212 non-null  float64
     39  ps_calc_01      595212 non-null  float64
     40  ps_calc_02      595212 non-null  float64
     41  ps_calc_03      595212 non-null  float64
     42  ps_calc_04      595212 non-null  int64  
     43  ps_calc_05      595212 non-null  int64  
     44  ps_calc_06      595212 non-null  int64  
     45  ps_calc_07      595212 non-null  int64  
     46  ps_calc_08      595212 non-null  int64  
     47  ps_calc_09      595212 non-null  int64  
     48  ps_calc_10      595212 non-null  int64  
     49  ps_calc_11      595212 non-null  int64  
     50  ps_calc_12      595212 non-null  int64  
     51  ps_calc_13      595212 non-null  int64  
     52  ps_calc_14      595212 non-null  int64  
     53  ps_calc_15_bin  595212 non-null  int64  
     54  ps_calc_16_bin  595212 non-null  int64  
     55  ps_calc_17_bin  595212 non-null  int64  
     56  ps_calc_18_bin  595212 non-null  int64  
     57  ps_calc_19_bin  595212 non-null  int64  
     58  ps_calc_20_bin  595212 non-null  int64  
    dtypes: float64(10), int64(49)
    memory usage: 267.9 MB


- 14개의 Categorical 변수(cat)는 더미 변수를 만들어야하고, binary 변수(bin)는 binary이기에 더미변수를 만들지 않아도 됩니다.
- 데이터는 float이거나 int64 데이터 타입입니다.
- Null값은 없는것으로 나오는데, 누락값은 -1로 처리하였기 때문입니다.

<br>

## 5. Defining the metadata
---
### 5.1 Metadata


```python
data = []
for f in train.columns:
    # Defining the role
    if f == 'target':
        role = 'target'
    elif f == 'id':
        role = 'id'
    else:
        role = 'input'

    # Defining the level
    if 'bin' in f or f == 'target':
        level = 'binary'
    elif 'cat' in f or f == 'id':
        level = 'nominal'
    elif train[f].dtype == float:
        level = 'interval'
    elif train[f].dtype == int:
        level = 'ordinal'
        
    # Initialize keep to True for all variables except for id
    
    keep = True
    if f == 'id':
        keep = False
        
    # Defining the data type
    
    dtype = train[f].dtype
    
    # Creating a Dict tha contains all the metadata for the variable
    
    f_dict = {
        'varname' : f,
        'role' : role,
        'level' : level,
        'keep' : keep,
        'dtype' : dtype
    }
    data.append(f_dict)
    
meta = pd.DataFrame(data, columns = ['varname', 'role', 'level', 'keep', 'dtype'])
meta.set_index('varname', inplace = True)
```


```python
meta
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
      <th>role</th>
      <th>level</th>
      <th>keep</th>
      <th>dtype</th>
    </tr>
    <tr>
      <th>varname</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>id</td>
      <td>nominal</td>
      <td>False</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>target</th>
      <td>target</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_01</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_02_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_03</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_04_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_05_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_06_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_07_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_08_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_09_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_10_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_11_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_12_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_13_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_14</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_15</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_16_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_17_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_18_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_reg_01</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_reg_02</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_reg_03</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_car_01_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_02_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_03_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_04_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_05_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_06_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_07_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_08_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_09_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_10_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_11_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_11</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_12</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_car_13</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_car_14</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_car_15</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_calc_01</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_calc_02</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_calc_03</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_calc_04</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_05</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_06</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_07</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_08</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_09</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_10</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_11</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_12</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_13</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_14</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_15_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_16_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_17_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_18_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_19_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_20_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
  </tbody>
</table>
</div>



- 데이터의 시각화, 분석, 모델링 위해 모델의 메타데이터를 데이터프레임에 저장함
    - role: input, ID, target
    - level: nominal, interval, ordinal, binary
    - keep: True or False
    - dtype: int, float, str
<br>


```python
meta[(meta.level == 'nominal') & (meta.keep == True)].index
```




    Index(['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat',
           'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat',
           'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat',
           'ps_car_10_cat', 'ps_car_11_cat'],
          dtype='object', name='varname')



- Kepp은 True(삭제되지않음)인 nominal 변수들의 목록

<br>


```python
pd.DataFrame({'count' : meta.groupby(['role', 'level'])['role'].size()}).reset_index()
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
      <th>role</th>
      <th>level</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id</td>
      <td>nominal</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>input</td>
      <td>binary</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>input</td>
      <td>interval</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>input</td>
      <td>nominal</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>input</td>
      <td>ordinal</td>
      <td>16</td>
    </tr>
    <tr>
      <th>5</th>
      <td>target</td>
      <td>binary</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



- Role 및 level별 변수의 count를 Group By로 묶어서 DataFrame으로 만듬

<br>

## 6. Descriptive statistics
---
### 6.1 Descriptive statistics

- 데이터프레임에 describe를 할수 있습니다. 다만, 범주형 범수에는 의미가 없으니 실수형 변수에 사용하여 평균, 표준편차 등을 알수 있습니다.

<br>

### 6.2 Interval variables


```python
v = meta[(meta.level == 'interval') & (meta.keep == True)].index
train[v].describe()
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
      <th>ps_reg_01</th>
      <th>ps_reg_02</th>
      <th>ps_reg_03</th>
      <th>ps_car_12</th>
      <th>ps_car_13</th>
      <th>ps_car_14</th>
      <th>ps_car_15</th>
      <th>ps_calc_01</th>
      <th>ps_calc_02</th>
      <th>ps_calc_03</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.610991</td>
      <td>0.439184</td>
      <td>0.551102</td>
      <td>0.379945</td>
      <td>0.813265</td>
      <td>0.276256</td>
      <td>3.065899</td>
      <td>0.449756</td>
      <td>0.449589</td>
      <td>0.449849</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.287643</td>
      <td>0.404264</td>
      <td>0.793506</td>
      <td>0.058327</td>
      <td>0.224588</td>
      <td>0.357154</td>
      <td>0.731366</td>
      <td>0.287198</td>
      <td>0.286893</td>
      <td>0.287153</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>0.250619</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.400000</td>
      <td>0.200000</td>
      <td>0.525000</td>
      <td>0.316228</td>
      <td>0.670867</td>
      <td>0.333167</td>
      <td>2.828427</td>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.700000</td>
      <td>0.300000</td>
      <td>0.720677</td>
      <td>0.374166</td>
      <td>0.765811</td>
      <td>0.368782</td>
      <td>3.316625</td>
      <td>0.500000</td>
      <td>0.400000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.900000</td>
      <td>0.600000</td>
      <td>1.000000</td>
      <td>0.400000</td>
      <td>0.906190</td>
      <td>0.396485</td>
      <td>3.605551</td>
      <td>0.700000</td>
      <td>0.700000</td>
      <td>0.700000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.900000</td>
      <td>1.800000</td>
      <td>4.037945</td>
      <td>1.264911</td>
      <td>3.720626</td>
      <td>0.636396</td>
      <td>3.741657</td>
      <td>0.900000</td>
      <td>0.900000</td>
      <td>0.900000</td>
    </tr>
  </tbody>
</table>
</div>



- Level이 interval인 변수에 대해 describe를 진행하였습니다.
- reg 변수들중에는 ps_reg_03에만 -1(Null data)가 있습니다.
- car 변수들중에는 ps_car_12, ps_car_14에 -1(Null data)가 있습니다.
- calc 변수들에는 -1(NUll data)는 없습니다.
- 변수별로 min과 max의 range가 다릅니다, 스케일링을 적용해야 할듯 합니다.
- interval 변수들의 범위는 그렇게 크지 않음을 알수 있습니다.

<br>

### 6.3 Ordinal variables


```python
v = meta[(meta.level == 'ordinal') & (meta.keep == True)].index
train[v].describe()
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
      <th>ps_ind_01</th>
      <th>ps_ind_03</th>
      <th>ps_ind_14</th>
      <th>ps_ind_15</th>
      <th>ps_car_11</th>
      <th>ps_calc_04</th>
      <th>ps_calc_05</th>
      <th>ps_calc_06</th>
      <th>ps_calc_07</th>
      <th>ps_calc_08</th>
      <th>ps_calc_09</th>
      <th>ps_calc_10</th>
      <th>ps_calc_11</th>
      <th>ps_calc_12</th>
      <th>ps_calc_13</th>
      <th>ps_calc_14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.900378</td>
      <td>4.423318</td>
      <td>0.012451</td>
      <td>7.299922</td>
      <td>2.346072</td>
      <td>2.372081</td>
      <td>1.885886</td>
      <td>7.689445</td>
      <td>3.005823</td>
      <td>9.225904</td>
      <td>2.339034</td>
      <td>8.433590</td>
      <td>5.441382</td>
      <td>1.441918</td>
      <td>2.872288</td>
      <td>7.539026</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.983789</td>
      <td>2.699902</td>
      <td>0.127545</td>
      <td>3.546042</td>
      <td>0.832548</td>
      <td>1.117219</td>
      <td>1.134927</td>
      <td>1.334312</td>
      <td>1.414564</td>
      <td>1.459672</td>
      <td>1.246949</td>
      <td>2.904597</td>
      <td>2.332871</td>
      <td>1.202963</td>
      <td>1.694887</td>
      <td>2.746652</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>2.000000</td>
      <td>8.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>8.000000</td>
      <td>3.000000</td>
      <td>9.000000</td>
      <td>2.000000</td>
      <td>8.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>9.000000</td>
      <td>4.000000</td>
      <td>10.000000</td>
      <td>3.000000</td>
      <td>10.000000</td>
      <td>7.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.000000</td>
      <td>11.000000</td>
      <td>4.000000</td>
      <td>13.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>6.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>12.000000</td>
      <td>7.000000</td>
      <td>25.000000</td>
      <td>19.000000</td>
      <td>10.000000</td>
      <td>13.000000</td>
      <td>23.000000</td>
    </tr>
  </tbody>
</table>
</div>



- ps_car_11 변수에만 -1(Null data)가 있습니다.
- 모두 min, max range가 다르므로 scaling을 진행해야 합니다.

<br>

### 6.4 Binary variables


```python
v = meta[(meta.level == 'binary') & (meta.keep == True)].index
train[v].describe()
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
      <th>target</th>
      <th>ps_ind_06_bin</th>
      <th>ps_ind_07_bin</th>
      <th>ps_ind_08_bin</th>
      <th>ps_ind_09_bin</th>
      <th>ps_ind_10_bin</th>
      <th>ps_ind_11_bin</th>
      <th>ps_ind_12_bin</th>
      <th>ps_ind_13_bin</th>
      <th>ps_ind_16_bin</th>
      <th>ps_ind_17_bin</th>
      <th>ps_ind_18_bin</th>
      <th>ps_calc_15_bin</th>
      <th>ps_calc_16_bin</th>
      <th>ps_calc_17_bin</th>
      <th>ps_calc_18_bin</th>
      <th>ps_calc_19_bin</th>
      <th>ps_calc_20_bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.036448</td>
      <td>0.393742</td>
      <td>0.257033</td>
      <td>0.163921</td>
      <td>0.185304</td>
      <td>0.000373</td>
      <td>0.001692</td>
      <td>0.009439</td>
      <td>0.000948</td>
      <td>0.660823</td>
      <td>0.121081</td>
      <td>0.153446</td>
      <td>0.122427</td>
      <td>0.627840</td>
      <td>0.554182</td>
      <td>0.287182</td>
      <td>0.349024</td>
      <td>0.153318</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.187401</td>
      <td>0.488579</td>
      <td>0.436998</td>
      <td>0.370205</td>
      <td>0.388544</td>
      <td>0.019309</td>
      <td>0.041097</td>
      <td>0.096693</td>
      <td>0.030768</td>
      <td>0.473430</td>
      <td>0.326222</td>
      <td>0.360417</td>
      <td>0.327779</td>
      <td>0.483381</td>
      <td>0.497056</td>
      <td>0.452447</td>
      <td>0.476662</td>
      <td>0.360295</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



- Train 데이터에서 target은 3.645% 입니다. 이것은 강력한 불균형 (strongly imbalanced) 입니다.
- 이것은 대부분의 값이 0으로 되어있음을 의미합니다.

<br>

## 7. Handling imbalanced classes
---
### 7.1 Handling imbalanced classes
- Target = 1인 Record의 비율이 너무 적습니다. 그 말인즉슨, 모두다 target을 0으로 예측해도 얼마안되는 1만 틀린것으로 파악됩니다.
- 이를 해결하기 위해 1을 오버 샘플링하거나 0을 언더샘플링 하는 방법이 있습니다.
- 이번에는 언더샘플링을 하겠습니다.

<br>

### 7.2 UnderSampling


```python
desired_apriori=0.10

# Get the indices per target value
idx_0 = train[train.target == 0].index
idx_1 = train[train.target == 1].index

# Get original number of records per target value
nb_0 = len(train.loc[idx_0])
nb_1 = len(train.loc[idx_1])

# Calculate the undersampling rate and resulting number of records with target=0
undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)
undersampled_nb_0 = int(undersampling_rate*nb_0)
print('Rate to undersample records with target=0: {}'.format(undersampling_rate))
print('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0))

# Randomly select records with target=0 to get at the desired a priori
undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_nb_0)

# Construct list with remaining indices
idx_list = list(undersampled_idx) + list(idx_1)

# Return undersample data frame
train = train.loc[idx_list].reset_index(drop=True)
under_rate = train['target'].sum() / train['target'].count()
print(f'Under sampling으로 변환된 target의 비율 : {under_rate} %')
```

    Rate to undersample records with target=0: 0.34043569687437886
    Number of records with target=0 after undersampling: 195246
    Under sampling으로 변환된 target의 비율 : 0.1 %


- undersampling_rate는 0인 타겟이 몇%가 되어야 target 1이 1%가 되는지의 대한 비율
- desired_apriori = 0.10 는 undersampling 하여 나오게될 target = 1의 비율

<br>

## 8. Data Quality Checks
---
### 8.1 Checking missing values


```python
vars_with_missing = []

for f in train.columns:
    missings = train[train[f] == -1][f].count()
    if missings > 0:
        vars_with_missing.append(f)
        missings_perc = missings / train.shape[0]
        
        print(f'Variable {f} has {missings} records {missings_perc:.2%} with missing values')
print(f'In total, there are {len(vars_with_missing)} varialbles with missing values')
```

    Variable ps_ind_02_cat has 103 records 0.05% with missing values
    Variable ps_ind_04_cat has 51 records 0.02% with missing values
    Variable ps_ind_05_cat has 2256 records 1.04% with missing values
    Variable ps_reg_03 has 38580 records 17.78% with missing values
    Variable ps_car_01_cat has 62 records 0.03% with missing values
    Variable ps_car_02_cat has 2 records 0.00% with missing values
    Variable ps_car_03_cat has 148367 records 68.39% with missing values
    Variable ps_car_05_cat has 96026 records 44.26% with missing values
    Variable ps_car_07_cat has 4431 records 2.04% with missing values
    Variable ps_car_09_cat has 230 records 0.11% with missing values
    Variable ps_car_11 has 1 records 0.00% with missing values
    Variable ps_car_14 has 15726 records 7.25% with missing values
    In total, there are 12 varialbles with missing values


- Missing Values(Null Data)인 -1을 각 변수별로 찾아서, 비율을 확인한것 입니다.
- 생각보다 Missing Values가 많은 변수가 있습니다. ps_res_03, ps_car_03_cat, ps_car_05_cat ...
- 총 12개의 변수에서 Missing values가 있습니다.

<br>


```python
# Dropping the variables with too many missing values
vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
train.drop(vars_to_drop, inplace=True, axis=1)
meta.loc[(vars_to_drop),'keep'] = False  # Updating the meta

# Imputing with the mean or mode
mean_imp = SimpleImputer(missing_values=-1, strategy='mean')
mode_imp = SimpleImputer(missing_values=-1, strategy='most_frequent')
train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()
train['ps_car_12'] = mean_imp.fit_transform(train[['ps_car_12']]).ravel()
train['ps_car_14'] = mean_imp.fit_transform(train[['ps_car_14']]).ravel()
train['ps_car_11'] = mode_imp.fit_transform(train[['ps_car_11']]).ravel()
```

- Missing value이있는 다른범주형 변수의 경우 Missing value -1을 그대로 둠
- ps_reg_03 (continuous)의 18%의 Missing value는 평균으로 바꿉니다.
- ps_car_11 (ordinal)의 5개의 Misisng values는 최빈값으로 바꿉니다.
- ps_car_12 (continuous)의 단 1개의 Missing value 평균으로 바꿉니다.
- ps_car_14 (continuous)의 7% Missing values는 평균으로 바꿉니다.

<br>

### 8.2 Checking the cardinality of the categorical variables


```python
v = meta[(meta['level'] == 'nominal') & (meta['keep'])].index

for f in v:
    dist_values = train[f].value_counts().shape[0]
    print(f'Variable {f} has {dist_values} distinct values')
```

    Variable ps_ind_02_cat has 5 distinct values
    Variable ps_ind_04_cat has 3 distinct values
    Variable ps_ind_05_cat has 8 distinct values
    Variable ps_car_01_cat has 13 distinct values
    Variable ps_car_02_cat has 3 distinct values
    Variable ps_car_04_cat has 10 distinct values
    Variable ps_car_06_cat has 18 distinct values
    Variable ps_car_07_cat has 3 distinct values
    Variable ps_car_08_cat has 2 distinct values
    Variable ps_car_09_cat has 6 distinct values
    Variable ps_car_10_cat has 3 distinct values
    Variable ps_car_11_cat has 104 distinct values


- cardinality는 변수에있는 서로 다른 값의 수를 나타냅니다. 
- 범주 형 변수에서 더미 변수를 만들 것이므로 고유 한 값이 많은 변수가 있는지 확인해야합니다. 이러한 변수는 많은 더미 변수를 생성하므로 다르게 처리해야합니다
- ps_car_11_cat이 104개의 distinct data를 가집니다.

<br>


```python
# Script by https://www.kaggle.com/ogrellier
# Code: https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
```


```python
train_encoded, test_encoded = target_encode(train['ps_car_11_cat'],
                                            test['ps_car_11_cat'],
                                            target=train.target,
                                            min_samples_leaf=100,
                                            smoothing=10,
                                            noise_level=0.01)

train['ps_car_11_cat_te'] = train_encoded
train.drop('ps_car_11_cat', axis=1, inplace=True)
meta.loc['ps_car_11_cat', 'keep'] = False  # Updating the meta
test['ps_car_11_cat_te'] = test_encoded
test.drop('ps_car_11_cat', axis=1, inplace=True)
```

<br>

## 9. Exploratory Data Visualization
---
### 9.1 Categorical variables


```python
v = meta[(meta.level == 'nominal') & (meta.keep)].index

for f in v:
    plt.figure()
    fig, ax = plt.subplots(figsize = (20,10))
    
    # Calculate the percentage of target = 1 per category value
    cat_perc = train[[f, 'target']].groupby([f], as_index = False).mean()
    cat_perc.sort_values(by = 'target', ascending = False, inplace = True)
    
    # Bar plot
    # Order the bars descending on target mean
    sns.barplot(ax = ax, x = f, y = 'target', data = cat_perc, order= cat_perc[f])
    plt.title(f'barplot of {f}', fontsize = 18)
    plt.ylabel('% Target', fontsize = 18)
    plt.xlabel(f, fontsize = 18)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 18)
    plt.show()
```


    <Figure size 432x288 with 0 Axes>

<img src = 'https://user-images.githubusercontent.com/60168331/98533692-a4484200-22c6-11eb-9fb3-f63c6111be71.png'>

    <Figure size 432x288 with 0 Axes>

<img src = 'https://user-images.githubusercontent.com/60168331/98533696-a5796f00-22c6-11eb-95be-f9052ebeb69a.png'>

    <Figure size 432x288 with 0 Axes>

<img src = 'https://user-images.githubusercontent.com/60168331/98533700-a6120580-22c6-11eb-85b0-d058aee3710b.png'>

    <Figure size 432x288 with 0 Axes>

<img src = 'https://user-images.githubusercontent.com/60168331/98533702-a6aa9c00-22c6-11eb-8894-5a2e2eb2d167.png'>

    <Figure size 432x288 with 0 Axes>

<img src = 'https://user-images.githubusercontent.com/60168331/98533703-a7433280-22c6-11eb-91d7-9bb990de917d.png'>

    <Figure size 432x288 with 0 Axes>

<img src = 'https://user-images.githubusercontent.com/60168331/98533705-a7433280-22c6-11eb-81c2-ba88415da9a8.png'>

    <Figure size 432x288 with 0 Axes>

<img src = 'https://user-images.githubusercontent.com/60168331/98533706-a7dbc900-22c6-11eb-8dfe-23cc366683b2.png'>

    <Figure size 432x288 with 0 Axes>

<img src = 'https://user-images.githubusercontent.com/60168331/98533707-a8745f80-22c6-11eb-93fc-b3ff612e01e2.png'>

    <Figure size 432x288 with 0 Axes>

<img src = 'https://user-images.githubusercontent.com/60168331/98533709-a8745f80-22c6-11eb-8660-ab551e8798ae.png'>

    <Figure size 432x288 with 0 Axes>

<img src = 'https://user-images.githubusercontent.com/60168331/98533711-a90cf600-22c6-11eb-9bd2-4e60cae88fe4.png'>

    <Figure size 432x288 with 0 Axes>

<img src = 'https://user-images.githubusercontent.com/60168331/98533713-a90cf600-22c6-11eb-94d7-7c1a40737fba.png'>


- Categorical variables와 Target = 1인 고객 비율을 살펴 보겠습니다.
- Missing value가 있는 변수에서 알 수 있듯이 Missing value를 다른 값으로 대체하는 대신 별도의 범주 값으로 유지하는 것이 좋습니다. 
- Missing value가 있는 고객은 보험 청구를 요청할 가능성이 훨씬 더 높은 (경우에 따라 훨씬 더 낮은) 것으로 보입니다

<br>

### 9.2 Interval variables


```python
def corr_heatmap(v):
    correlations = train[v].corr()

    # Create color map ranging between two colors
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .75})
    plt.show();
    
v = meta[(meta.level == 'interval') & (meta.keep)].index
corr_heatmap(v)
```

<img src = 'https://user-images.githubusercontent.com/60168331/98533714-a9a58c80-22c6-11eb-8f82-4e10229f63f2.png'>


- Interval variables 간의 상관 관계를 확인합니다. 
- heatmap은 변수 간의 상관 관계를 시각화하는 좋은 방법입니다.
- 아래의 변수들은 강한 상관 관계를 가집니다.
    - ps_reg_02 and ps_reg_03 (0.7)
    - ps_car_12 and ps_car13 (0.67)
    - ps_car_12 and ps_car14 (0.58)
    - ps_car_13 and ps_car15 (0.67)
- Seaborn은 변수들 사이의 (선형) 관계를 시각화할 수 있는 몇 가지 유용한 플롯을 가지고 있다. 우리는 변수들 사이의 관계를 시각화하기 위해 Pairplot 사용할 수 있습니다.
- 하지만 Heatmap에서 이미 제한된 수의 상관 변수를 보여 주었기 때문에, 우리는 각각의 높은 상관 관계를 가진 변수들을 개별적으로 살펴보도록 하겠습니다.

<br>


```python
s = train.sample(frac = 0.1)
```

- 참고: 속도를 높이기 위해 학습 데이터의 샘플을 가져옵니다.

<br>


```python
sns.lmplot(x='ps_reg_02', y='ps_reg_03', data=s, hue='target',
           palette='Set1', scatter_kws={'alpha': 0.3})
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98533716-a9a58c80-22c6-11eb-9a0a-46c9efaf671a.png'>


- ps_reg_02 및 ps_reg_03 회귀선에서 알 수 있듯이 이러한 변수 사이에는 선형 관계가 있습니다. 
- hue 매개 변수는 target = 0과 target = 1에 대한 회귀선이 동일 함을 알 수 있습니다.

<br>


```python
sns.lmplot(x='ps_car_12', y='ps_car_13', data=s, hue='target',
           palette='Set1', scatter_kws={'alpha': 0.3})
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98533718-aa3e2300-22c6-11eb-9a00-5657932f2ef5.png'>


- ps_car_12, ps_car_13의 선형관계

<br>


```python
sns.lmplot(x='ps_car_12', y='ps_car_14', data=s, hue='target',
           palette='Set1', scatter_kws={'alpha': 0.3})
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98533719-aad6b980-22c6-11eb-8f76-e9ddc16860a6.png'>


- ps_car_12, ps_car_14의 선형관계

<br>


```python
sns.lmplot(x='ps_car_15', y='ps_car_13', data=s, hue='target',
           palette='Set1', scatter_kws={'alpha': 0.3})
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98533720-aad6b980-22c6-11eb-8f51-88f97c289088.png'>


- ps_car_15와 ps_car_13

<br>

- 변수에 대해 PCA (주성분 분석)를 수행하여 차원을 줄일 수 있습니다.
- 하지만 상관 변수의 수가 적기 때문에 모델이 무거운 작업을 수행하도록 할 것입니다.

<br>

### 9.3 Checking the correlations betwwen ordinal variables


```python
v = meta[(meta.level == 'ordinal') & (meta.keep)].index
corr_heatmap(v)
```

<img src = 'https://user-images.githubusercontent.com/60168331/98533721-ab6f5000-22c6-11eb-876d-4596468499fc.png'>


- Ordinal variables의 경우 많은 상관 관계를 볼 수 없다. 
- 하지만 Tatget Value으로 그룹화할 때 분포가 어떻게 되는지 살펴볼 수 있다.

<br>

## 10. Feature engineering
---
### 10.1 Creating dummy variables


```python
v = meta[(meta.level == 'nominal') & (meta.keep)].index
print(f'Before dummification we have {train.shape[1]} variables in train.')
train = pd.get_dummies(train, columns= v, drop_first= True)
print(f'After dummification we have {train.shape[1]} variables in train.')
```

    Before dummification we have 57 variables in train.
    After dummification we have 109 variables in train.


- Categorical variables의 값은 순서나 크기를 나타내지 않는다. 예를 들어 범주 2는 범주 1의 두 배가 아니다. 
- 그러므로 우리는 그것을 다룰 더미 변수를 만들 수 있다. 
- 이 정보는 원래 변수의 범주에 대해 생성된 다른 더미 변수에서 파생될 수 있으므로 첫 번째 더미 변수를 삭제한다.
- 총 52개의 dummy 변수를 생성하였습니다.

<br>

### 10.2 Creating interaction variables


```python
v = meta[(meta.level == 'interval') & (meta.keep)].index

poly = PolynomialFeatures(degree = 2, interaction_only= False, include_bias= False)
interactions = pd.DataFrame(data = poly.fit_transform(train[v]), columns=poly.get_feature_names(v))
interactions.drop(v, axis = 1, inplace = True) # Remove the original columns

# Concat the interaction variables to the train data
print(f'Before creating interactions we have {train.shape[1]} variables in train.')

train = pd.concat([train, interactions], axis = 1)

print(f'After creating interactions we have {train.shape[1]} variables in train.')
```

    Before creating interactions we have 109 variables in train.
    After creating interactions we have 164 variables in train.


- get_feature_names 메서드를 사용해서 편하게 interactions variables을 추가하였습니다.

<br>
    
## 11. Feature selection
---
### 11.1 Removing features with low or zero variance


```python
selector = VarianceThreshold(threshold=0.01)
selector.fit(train.drop(['id', 'target'], axis = 1)) # Fit to train without id and target variables

f = np.vectorize(lambda x : not x) # Function to toggle boolean_array elements
v = train.drop(['id', 'target'], axis = 1).columns[f(selector.get_support())]
print(f'{len(v)} variables have too low variance.')
print(f'These variables are {list(v)}')
```

    28 variables have too low variance.
    These variables are ['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_car_12', 'ps_car_14', 'ps_car_11_cat_te', 'ps_ind_05_cat_2', 'ps_ind_05_cat_5', 'ps_car_01_cat_1', 'ps_car_01_cat_2', 'ps_car_04_cat_3', 'ps_car_04_cat_4', 'ps_car_04_cat_5', 'ps_car_04_cat_6', 'ps_car_04_cat_7', 'ps_car_06_cat_2', 'ps_car_06_cat_5', 'ps_car_06_cat_8', 'ps_car_06_cat_12', 'ps_car_06_cat_16', 'ps_car_06_cat_17', 'ps_car_09_cat_4', 'ps_car_10_cat_1', 'ps_car_10_cat_2', 'ps_car_12^2', 'ps_car_12 ps_car_14', 'ps_car_14^2']


- 변동이 없거나 매우 낮은 특성을 제거하는 것입니다. (분산이 0인것)
- Sklearn에는 VarianceThreshold라는 편리한 방법이 있습니다. 기본적으로 분산이 0 인 기능을 제거합니다. 
- 이전 단계에서 0 분산 변수가 없음을 확인 했으므로이 대회에는 적용되지 않습니다. 
- 그러나 분산이 1 % 미만인 특성을 제거하면 31 개의 변수가 제거됩니다.
- 분산을 기반으로 선택하면 다소 많은 변수(31개)를 잃게됩니다. 그러나 변수가 너무 많지 않기 때문에 classifier가 선택하도록 할 것입니다. 
- 더 많은 변수가있는 데이터 세트의 경우 처리 시간을 줄일 수 있습니다.
- Sklearn은 또한 다른 기능 선택 방법과 함께 제공됩니다. 
- 이러한 메서드 중 하나는 another classifier가 최상의 기능을 선택하고 계속 진행하도록하는 SelectFromModel입니다. 
- 아래에서는 Random Forest로 수행하겠습니다.

<br>

### 11.2 Selecting features with a Random Forest and SelectFromModel


```python
X_train = train.drop(['id', 'target'], axis = 1)
y_train = train['target']

feat_labels = X_train.columns

rf = RandomForestClassifier(n_estimators= 1000, random_state= 0, n_jobs= -1)
rf.fit(X_train, y_train)
importances = rf.feature_importances_

indices = np.argsort(rf.feature_importances_)[::-1]

for f in range(X_train.shape[1]):
    print('%2d) %-*s %f' % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
```

     1) ps_car_11_cat_te               0.021062
     2) ps_car_13^2                    0.017319
     3) ps_car_13                      0.017288
     4) ps_car_12 ps_car_13            0.017244
     5) ps_car_13 ps_car_14            0.017148
     6) ps_reg_03 ps_car_13            0.017067
     7) ps_car_13 ps_car_15            0.016812
     8) ps_reg_01 ps_car_13            0.016788
     9) ps_reg_03 ps_car_14            0.016261
    10) ps_reg_03 ps_car_12            0.015580
    11) ps_reg_03 ps_car_15            0.015165
    12) ps_car_14 ps_car_15            0.015012
    13) ps_car_13 ps_calc_01           0.014751
    14) ps_car_13 ps_calc_03           0.014726
    15) ps_car_13 ps_calc_02           0.014673
    16) ps_reg_02 ps_car_13            0.014671
    17) ps_reg_01 ps_reg_03            0.014666
    18) ps_reg_01 ps_car_14            0.014455
    19) ps_reg_03^2                    0.014283
    20) ps_reg_03                      0.014255
    21) ps_reg_03 ps_calc_02           0.013804
    22) ps_reg_03 ps_calc_03           0.013758
    23) ps_reg_03 ps_calc_01           0.013711
    24) ps_calc_10                     0.013696
    25) ps_car_14 ps_calc_02           0.013633
    26) ps_car_14 ps_calc_01           0.013542
    27) ps_car_14 ps_calc_03           0.013499
    28) ps_calc_14                     0.013363
    29) ps_car_12 ps_car_14            0.012968
    30) ps_ind_03                      0.012923
    31) ps_car_14                      0.012806
    32) ps_car_14^2                    0.012734
    33) ps_reg_02 ps_car_14            0.012671
    34) ps_calc_11                     0.012585
    35) ps_reg_02 ps_reg_03            0.012559
    36) ps_ind_15                      0.012153
    37) ps_car_12 ps_car_15            0.010944
    38) ps_car_15 ps_calc_03           0.010888
    39) ps_car_15 ps_calc_02           0.010879
    40) ps_car_15 ps_calc_01           0.010851
    41) ps_calc_13                     0.010479
    42) ps_car_12 ps_calc_01           0.010467
    43) ps_car_12 ps_calc_03           0.010340
    44) ps_car_12 ps_calc_02           0.010287
    45) ps_reg_02 ps_car_15            0.010213
    46) ps_reg_01 ps_car_15            0.010201
    47) ps_calc_02 ps_calc_03          0.010092
    48) ps_calc_01 ps_calc_03          0.010010
    49) ps_calc_01 ps_calc_02          0.010005
    50) ps_calc_07                     0.009837
    51) ps_calc_08                     0.009801
    52) ps_reg_01 ps_car_12            0.009480
    53) ps_reg_02 ps_calc_01           0.009281
    54) ps_reg_02 ps_car_12            0.009270
    55) ps_reg_02 ps_calc_03           0.009218
    56) ps_reg_02 ps_calc_02           0.009210
    57) ps_reg_01 ps_calc_03           0.009043
    58) ps_reg_01 ps_calc_01           0.009036
    59) ps_calc_06                     0.009021
    60) ps_reg_01 ps_calc_02           0.008985
    61) ps_calc_09                     0.008808
    62) ps_ind_01                      0.008519
    63) ps_calc_05                     0.008296
    64) ps_calc_04                     0.008122
    65) ps_calc_12                     0.008066
    66) ps_reg_01 ps_reg_02            0.008024
    67) ps_car_15^2                    0.006172
    68) ps_car_15                      0.006147
    69) ps_calc_01                     0.005971
    70) ps_calc_03^2                   0.005967
    71) ps_calc_03                     0.005955
    72) ps_calc_02                     0.005949
    73) ps_calc_01^2                   0.005949
    74) ps_calc_02^2                   0.005930
    75) ps_car_12                      0.005373
    76) ps_car_12^2                    0.005366
    77) ps_reg_02^2                    0.005007
    78) ps_reg_02                      0.004993
    79) ps_reg_01                      0.004152
    80) ps_reg_01^2                    0.004116
    81) ps_car_11                      0.003787
    82) ps_ind_05_cat_0                0.003570
    83) ps_ind_17_bin                  0.002847
    84) ps_calc_17_bin                 0.002692
    85) ps_calc_16_bin                 0.002611
    86) ps_calc_19_bin                 0.002534
    87) ps_calc_18_bin                 0.002485
    88) ps_ind_16_bin                  0.002397
    89) ps_ind_04_cat_0                0.002387
    90) ps_car_01_cat_11               0.002376
    91) ps_ind_04_cat_1                0.002370
    92) ps_ind_07_bin                  0.002327
    93) ps_car_09_cat_2                0.002292
    94) ps_ind_02_cat_1                0.002249
    95) ps_car_09_cat_0                0.002115
    96) ps_car_01_cat_7                0.002103
    97) ps_ind_02_cat_2                0.002093
    98) ps_calc_20_bin                 0.002081
    99) ps_ind_06_bin                  0.002042
    100) ps_calc_15_bin                 0.001985
    101) ps_car_06_cat_1                0.001983
    102) ps_car_07_cat_1                0.001971
    103) ps_ind_08_bin                  0.001952
    104) ps_car_09_cat_1                0.001833
    105) ps_car_06_cat_11               0.001810
    106) ps_ind_09_bin                  0.001731
    107) ps_ind_18_bin                  0.001718
    108) ps_car_01_cat_10               0.001593
    109) ps_car_01_cat_9                0.001580
    110) ps_car_06_cat_14               0.001549
    111) ps_car_01_cat_6                0.001547
    112) ps_car_01_cat_4                0.001545
    113) ps_ind_05_cat_6                0.001502
    114) ps_ind_02_cat_3                0.001437
    115) ps_car_07_cat_0                0.001388
    116) ps_car_08_cat_1                0.001345
    117) ps_car_01_cat_8                0.001335
    118) ps_car_02_cat_1                0.001329
    119) ps_car_02_cat_0                0.001314
    120) ps_car_06_cat_4                0.001232
    121) ps_ind_05_cat_4                0.001212
    122) ps_car_01_cat_5                0.001151
    123) ps_ind_02_cat_4                0.001149
    124) ps_car_06_cat_6                0.001111
    125) ps_car_06_cat_10               0.001066
    126) ps_ind_05_cat_2                0.001025
    127) ps_car_04_cat_1                0.001017
    128) ps_car_06_cat_7                0.000991
    129) ps_car_04_cat_2                0.000979
    130) ps_car_01_cat_3                0.000899
    131) ps_car_09_cat_3                0.000879
    132) ps_car_01_cat_0                0.000872
    133) ps_car_06_cat_15               0.000851
    134) ps_ind_14                      0.000846
    135) ps_car_06_cat_9                0.000796
    136) ps_ind_05_cat_1                0.000740
    137) ps_car_06_cat_3                0.000706
    138) ps_car_10_cat_1                0.000700
    139) ps_ind_12_bin                  0.000689
    140) ps_ind_05_cat_3                0.000671
    141) ps_car_09_cat_4                0.000631
    142) ps_car_01_cat_2                0.000562
    143) ps_car_04_cat_8                0.000561
    144) ps_car_06_cat_17               0.000511
    145) ps_car_06_cat_16               0.000481
    146) ps_car_04_cat_9                0.000433
    147) ps_car_06_cat_12               0.000422
    148) ps_car_06_cat_13               0.000385
    149) ps_car_01_cat_1                0.000379
    150) ps_ind_05_cat_5                0.000305
    151) ps_car_06_cat_5                0.000283
    152) ps_ind_11_bin                  0.000218
    153) ps_car_04_cat_6                0.000207
    154) ps_ind_13_bin                  0.000148
    155) ps_car_04_cat_3                0.000146
    156) ps_car_06_cat_2                0.000137
    157) ps_car_06_cat_8                0.000099
    158) ps_car_04_cat_5                0.000098
    159) ps_car_04_cat_7                0.000082
    160) ps_ind_10_bin                  0.000072
    161) ps_car_10_cat_2                0.000062
    162) ps_car_04_cat_4                0.000044


- 여기서는 랜덤 포레스트의 feature importances를 기준으로 기능 선택을 할 것입니다. 
- Sklearn의 SelectFromModel을 사용하면 유지할 변수 수를 지정할 수 있습니다. 
- feature importances 수준에 대한 threshold를 수동으로 설정할 수 있습니다. 
- 그러나 우리는 단순히 상위 50 % 최고의 변수를 선택합니다.
- 위의 코드는 Sebastian Raschka의 GitHub 저장소에서 가져 왔습니다.

<br>


```python
sfm = SelectFromModel(rf, threshold='median', prefit=True)
print(f'Number of features before selection : {X_train.shape[1]}')

n_features = sfm.transform(X_train).shape[1]
print(f'Number of features after selection : {n_features}')
selected_vars = list(feat_labels[sfm.get_support()])
```

    Number of features before selection : 162
    Number of features after selection : 81


- SelectFromModel을 사용하여 사용할 prefit classifier와 feature importances에 대한 threshold을 지정할 수 있습니다. 
- get_support 메소드를 사용하면 train 데이터의 변수 수를 제한 할 수 있습니다.

<br>


```python
train = train[selected_vars + ['target']]
```

- train 데이터에 target까지 더함

<br>

## 12. Feature scaling
---
### 12.1 Feature scaling


```python
scaler = StandardScaler()
scaler.fit_transform(train.drop(['target'], axis = 1))
```




    array([[-0.45941104, -1.26665356,  1.05087653, ..., -0.72553616,
            -1.01071913, -1.06173767],
           [ 1.55538958,  0.95034274, -0.63847299, ..., -1.06120876,
            -1.01071913,  0.27907892],
           [ 1.05168943, -0.52765479, -0.92003125, ...,  1.95984463,
            -0.56215309, -1.02449277],
           ...,
           [-0.9631112 ,  0.58084336,  0.48776003, ..., -0.46445747,
             0.18545696,  0.27907892],
           [-0.9631112 , -0.89715418, -1.48314775, ..., -0.91202093,
            -0.41263108,  0.27907892],
           [-0.45941104, -1.26665356,  1.61399304, ...,  0.28148164,
            -0.11358706, -0.72653353]])



- train 데이터에 standardscaler를 적용 할 수 있습니다. 
- 이 작업이 완료되면 일부 classifier가 더 잘 작동됩니다.

<br>

## 13. Conclusion
---
### 13.1 Conclusion
- Porto Seguro Safe Driver Prediction의 EDA Note book
- Kaggle 필사를 진행 한것입니다.
