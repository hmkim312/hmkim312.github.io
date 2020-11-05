---
title: Prediction을 위한 Titanic EDA
author: HyunMin Kim
date: 2020-11-05 00:30:00 0000
categories: [Kaggle, Titanic]
tags: [Kaggle Transcription, Titanic, EDA, Sklearn]
---

- The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. That's why the name DieTanic. This is a very unforgetable disaster that no one in the world can forget.

- It took about $7.5 million to build the Titanic and it sunk under the ocean due to collision. The Titanic Dataset is a very good dataset for begineers to start a journey in data science and participate in competitions in Kaggle.

- The Objective of this notebook is to give an idea how is the workflow in any predictive modeling problem. How do we check features, how do we add new features and some Machine Learning Concepts. I have tried to keep the notebook as basic as possible so that even newbies can understand every phase of it.

- If You Like the notebook and think that it helped you..PLEASE UPVOTE. It will keep me motivated.

- 출처 : <https://www.kaggle.com/ash316/eda-to-prediction-dietanic>{:target="_blank"}

### Contents of the Notebook:
#### Part1: Exploratory Data Analysis(EDA):
- 1)Analysis of the features.
- 2)Finding any relations or trends considering multiple features.

#### Part2: Feature Engineering and Data Cleaning:
- 1)Adding any few features.
- 2)Removing redundant features.
- 3)Converting features into suitable form for modeling.

#### Part3: Predictive Modeling
- 1)Running Basic Algorithms.
- 2)Cross Validation.
- 3)Ensembling.
- 4)Important Features Extraction.

---
<br>

### Part1: Exploratory Data Analysis(EDA)

```python
# 기본설정
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
```


```python
# 데이터 로드
data = pd.read_csv('data/train.csv')
```


```python
# 데이터 확인
data.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>


```python
# null 값 확인
# age와 cabin에 null값이 있음
data.isnull().sum()
```

    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64


<br>

#### How many Survived


```python
f, ax = plt.subplots(1, 2, figsize=(18, 8))
data['Survived'].value_counts().plot.pie(explode = [0,0.1], autopct = '%1.1f%%', ax = ax[0], shadow = True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data = data, ax = ax[1])
ax[1].set_title('Survived')
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234590-05f56d00-1fa4-11eb-987d-bb470a322e44.png'>

- Observations
    - 891명의 승객중 350명이 살았으며, 이는 38.4%에 해당한다.
    - 누가 살고 죽었는지 확인하기 위해 Sex, Port of Embarcation, Age, etc를 살펴보겠다

<br>

#### Analysing The Features

- Sex--> Categorical Feature


```python
data.groupby(['Sex', 'Survived'])['Survived'].count()
```




    Sex     Survived
    female  0            81
            1           233
    male    0           468
            1           109
    Name: Survived, dtype: int64




```python
f, ax = plt.subplots(1, 2, figsize=(18, 8))
data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar(ax = ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex', hue = 'Survived', data = data, ax = ax[1])
ax[1].set_title('Sex : Survived vs Dead')
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234600-0aba2100-1fa4-11eb-8c27-6c187831ad2e.png'>


- Observations
    - 남자보다 여자가 생존율이 2배 이상 좋음

- Pclass -> Ordinal Feature


```python
pd.crosstab(data.Pclass, data.Survived, margins=True).style.background_gradient(cmap = 'summer_r')
```

<img src = 'https://user-images.githubusercontent.com/60168331/98235160-cb400480-1fa4-11eb-96bd-fd5c03f20b41.png'>

```python
f, ax = plt.subplots(1,2,figsize = (18,8))
data['Pclass'].value_counts().plot.bar(color=['#CD7F32', '#FFDF00', '#D3D3D3'], ax = ax[0])
ax[0].set_title('Number Of Passengers By Pclass')
ax[0].set_ylabel('Count')
sns.countplot('Pclass', hue = 'Survived', data = data, ax= ax[1])
ax[1].set_title('Pclass : Survived vs Dead')
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234605-0beb4e00-1fa4-11eb-8f3a-77064f328ea2.png'>


- Observations
    - pclass가 낮을수록 생존율이 높음을 알수있다


```python
# 성별과 pclass를 같이 살펴보기
pd.crosstab([data.Sex, data.Survived], data.Pclass, margins=True).style.background_gradient(cmap = 'summer_r')
```

<img src = 'https://user-images.githubusercontent.com/60168331/98235229-e90d6980-1fa4-11eb-9609-71f8f95689f7.png'>


```python
sns.factorplot('Pclass', 'Survived', hue = 'Sex', data = data)
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234607-0d1c7b00-1fa4-11eb-8657-988c4205e7b6.png'>


- Observations
    - Pclass 또한 생존율에 영향을 미친 중요한 feature이다

- Age -> Continous Feature


```python
print('Oldset Passenger was of:', data['Age'].max(), 'Years')
print('Youngest Passenger was of:', data['Age'].min(), 'Years')
print('Average Age on the ship:', data['Age'].mean(), 'Years')
```

    Oldset Passenger was of: 80.0 Years
    Youngest Passenger was of: 0.42 Years
    Average Age on the ship: 29.69911764705882 Years



```python
f, ax = plt.subplots(1, 2, figsize=(18, 8))
sns.violinplot('Pclass', 'Age', hue='Survived',
               data=data, split=True, ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0, 110, 10))
sns.violinplot('Sex', 'Age', hue='Survived', data=data, split=True, ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0, 110, 10))
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234612-0db51180-1fa4-11eb-9827-4a203aed7d28.png'>


- Observations
    - 10세 미만 어린이는 생존율이 높고, 20~50대는 1등급을 제외하곤 모두 생존율이 낮다.


```python
# 이름앞에 mr mrs 등을 추출
data['Initial'] = 0
for i in data:
    data['Initial'] = data.Name.str.extract('([A-Za-z]+)\.')
data['Initial']
```

    0        Mr
    1       Mrs
    2      Miss
    3       Mrs
    4        Mr
           ... 
    886     Rev
    887    Miss
    888    Miss
    889      Mr
    890      Mr
    Name: Initial, Length: 891, dtype: object




```python
pd.crosstab(data.Initial, data.Sex).T.style.background_gradient(cmap='summer_r')
```

<img src = 'https://user-images.githubusercontent.com/60168331/98235539-443f5c00-1fa5-11eb-9bd7-09d7283837dd.png'>

```python
# miss, mr, mrs, other로 변경
data['Initial'].replace(['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don'], [
                        'Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other', 'Other', 'Other', 'Mr', 'Mr', 'Mr'], inplace=True)
```


```python
# 칭호별 평균 나이
data.groupby('Initial')['Age'].mean()
```

    Initial
    Master     4.574167
    Miss      21.860000
    Mr        32.739609
    Mrs       35.981818
    Other     45.888889
    Name: Age, dtype: float64




```python
# NaN Age 채우기
data.loc[(data.Age.isnull())&(data.Initial == 'Mr'), 'Age'] = 33
data.loc[(data.Age.isnull())&(data.Initial == 'Master'), 'Age'] = 5
data.loc[(data.Age.isnull())&(data.Initial == 'Miss'), 'Age'] = 22
data.loc[(data.Age.isnull())&(data.Initial == 'Mrs'), 'Age'] = 36
data.loc[(data.Age.isnull())&(data.Initial == 'Other'), 'Age'] = 46
```


```python
# Nan 확인 any()를 사용, Null이 한개라도 있으면 True를 반환
data.Age.isnull().any()
```
    False




```python
f, ax = plt.subplots(1,2,figsize = (20,10))
data[data.Survived == 0].Age.plot.hist(ax = ax[0], bins = 20, edgecolor = 'black', color = 'red')
ax[0].set_title('Survived = 0')
x1 = list(range(0,85,5))
ax[0].set_xticks(x1)

data[data.Survived == 1].Age.plot.hist(ax = ax[1], color = 'green', bins = 20, edgecolor = 'black')
ax[1].set_title('Survived = 1')
x2 = list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234620-0f7ed500-1fa4-11eb-8933-f547eafc65f8.png'>

- Observations
    - 5세 미만 아이는 살았다
    - 80세 노인도 살았다
    - 가장 많이 죽은 나이는 30-40대


```python
sns.factorplot('Pclass', 'Survived', col = 'Initial', data = data)
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234623-10176b80-1fa4-11eb-9dde-8bba945b01d2.png'>

- Observations
    - 객실 등급을 무시하고 어린이와 아이가 가장 먼저 구조되었다

- Embarked -> Categorical Value


```python
pd.crosstab([data.Embarked, data.Pclass], [data.Sex, data.Survived], margins=True).style.background_gradient(cmap = 'summer_r')
```

<img src = 'https://user-images.githubusercontent.com/60168331/98235718-8d8fab80-1fa5-11eb-8b31-100c9f1f4dc4.png'>


```python
sns.factorplot('Embarked','Survived', data = data)
fig = plt.gcf()
fig.set_size_inches(5,3)
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234624-10b00200-1fa4-11eb-8278-c9634fc50d12.png'>

```python
f, ax = plt.subplots(2, 2, figsize=(20, 15))
sns.countplot('Embarked', data=data, ax=ax[0, 0])
ax[0, 0].set_title('No Of Passengers Boarded')
sns.countplot('Embarked', hue='Sex', data=data, ax=ax[0, 1])
ax[0, 1].set_title('Male - Female Split for Embarked')
sns.countplot('Embarked', hue='Survived', data=data, ax=ax[1, 0])
ax[1, 0].set_title('Embarked vs Survived')
sns.countplot('Embarked', hue = 'Pclass', data = data, ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')
plt.subplots_adjust(wspace = 0.2, hspace=0.5)
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234626-10b00200-1fa4-11eb-8815-7fda3903d57b.png'>

- Observations
    - S 항구에서 많이 탔고 대부분 3등급객실임
    - C 항구는 1등급과 2등급 승객을 구조한것으로 예상됨
    - S 항구에서는 1등급 객실 인원이 많이 탔음
    - Q 항구는 95%의 승객이 3등급인원임


```python
sns.factorplot('Pclass', 'Survived', hue = 'Sex', col = 'Embarked', data = data)
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234629-11489880-1fa4-11eb-85c2-bd4c5669923d.png'>


- Observations
    - 1)Pclass1과 Pclass2의 여성은 Pclass와 관계없이 생존 확률은 거의 1이다.
    - 2)남녀 모두의 생존율이 매우 낮기 때문에 Pclass3 Passenger에게는 포트S가 매우 불행해 보인다.(돈 문제)
    - 3)포트 Q는 거의 모두가 Pclass 3에서 온 것처럼 남자에게는 가장 어울리지 않는 것 같다.

- Filling Embarked NaN


```python
# Embarked NaN은 갯수가 가장 많은 S로 바꿈
data['Embarked'].fillna('S', inplace = True)
```


```python
#isnull().any()로 null값 확인 any로 인해 True가 나오면 1개라도 Null이 있는것
data['Embarked'].isnull().any()
```

    False



- SibSip -> Discrete Feature
    - Sibling = brother, sister, stepbrother, stepsister


```python
pd.crosstab([data.SibSp], data.Survived).style.background_gradient(cmap = 'summer_r')
```

<img src = 'https://user-images.githubusercontent.com/60168331/98235907-d6dffb00-1fa5-11eb-8216-2efb6b0b0e41.png'>


```python
f, ax = plt.subplots(1, 2, figsize=(20, 8))
sns.barplot('SibSp', 'Survived', data=data, ax=ax[0])
ax[0].set_title('SibSp vs Survived')
sns.factorplot('SibSp', 'Survived', data=data, ax=ax[1])
ax[1].set_title('SibSp vs Survived')
# plt.close(2)
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234631-11e12f00-1fa4-11eb-80a8-6100d33f7dac.png'>
<img src = 'https://user-images.githubusercontent.com/60168331/98234632-1279c580-1fa4-11eb-8973-2995844a70e0.png'>


```python
pd.crosstab(data.SibSp, data.Pclass).style.background_gradient(cmap = 'summer_r')
```

<img src = 'https://user-images.githubusercontent.com/60168331/98236016-055dd600-1fa6-11eb-9846-1cffc4676908.png'>


- Observations
    - barplot과 factorplot에서 보이듯이 혼자온 탑승객은 34.5%의 생존율을 보임
    - 대가족은 3클래스에 많이 있어서 많이 죽음

- Parch


```python
pd.crosstab(data.Parch, data.Pclass).style.background_gradient(cmap = 'summer_r')
```

<img src = 'https://user-images.githubusercontent.com/60168331/98236095-23c3d180-1fa6-11eb-852f-dceb34395ae4.png'>


```python
sns.barplot('Parch', 'Survived', data = data)
plt.title('Parch vs Survived')

sns.factorplot('Parch', 'Survived', data = data)
plt.title('Parch vs Survived')
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234635-1279c580-1fa4-11eb-9a44-df716fa5f6d4.png'>
<img src = 'https://user-images.githubusercontent.com/60168331/98234637-13125c00-1fa4-11eb-82ee-ce11f57615af.png'>

- Observations
    - 1 ~ 3명의 부모를 둔 가족의 생존율이 높고, 그 이상은 생존율이 작아진다.

- Fare -> continous Feature


```python
print('Highest Fare was:', data['Fare'].max())
print('Lowest Fare was:', data['Fare'].min())
print('Average Fare was:', data['Fare'].mean())
```

    Highest Fare was: 512.3292
    Lowest Fare was: 0.0
    Average Fare was: 32.2042079685746



```python
f, ax = plt.subplots(1,3, figsize = (20,8))
sns.distplot(data[data['Pclass'] == 1].Fare, ax = ax[0])
ax[0].set_title('Fare in Pclass 1')

sns.distplot(data[data['Pclass'] == 2].Fare, ax = ax[1])
ax[1].set_title('Fare in Pclass 2')

sns.distplot(data[data['Pclass'] == 3].Fare, ax = ax[2])
ax[2].set_title('Fare in Pclass 3')

plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234640-13aaf280-1fa4-11eb-9390-10a77484cb1f.png'>

- Observations in a Nutshell for all features:
    - Sex: 여자가 남자보다 생존율이 높음
    - Pclass: 1등급객실이면 생존율이 높음
    - Age: 15 - 35 살 그룹 대비 5-10세 아이들의 생존율이 높다
    - Embarked: Pclass1 승객 대다수가 S에서 일어났음에도 불구하고 C에서 생존할 가능성은 더 좋아 보인다.
    - Parch+SibSp: 1-2명의 형제자매, 배우자 또는 1-3명의 부모들이 혼자 있거나 대가족이 당신과 함께 여행하는 것보다 생존할 가능성이 더 높다.

<br>

#### Correlation Between The Features


```python
sns.heatmap(data.corr(), annot= True, cmap = 'RdYlGn', linewidths=0.2)
fig = plt.gcf()
fig.set_size_inches(10,8)
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234643-14438900-1fa4-11eb-88ee-c022ebcb79c8.png'>

- interpreting The Heatmap
    - Pclass, Fare와 상관관계가 있음 
---
<br>

### Part2: Feature Engineering and Data Cleaning
- Feature의 중복을 제거하거나, 새로운 Feature를 만들어내는것
- 이름에서 Initail을 만들어내는 것과 같은것

<br>

#### Age_band
- 나이 변수는 지속적인 변수이며 이는 머신러닝에 적합하지 않음
- Age_band를 만들어 그룹화 시켜줄것임
- max age가 80세이고 5개로 그룹화하니 80/5 = 16이므로 16으로 나눔


```python
data['Age_band'] = 0
data.loc[data['Age'] <= 16, 'Age_band'] = 0
data.loc[(data['Age'] > 16) & (data['Age'] <=32), 'Age_band'] = 1
data.loc[(data['Age'] > 32) & (data['Age'] <=48), 'Age_band'] = 2
data.loc[(data['Age'] > 48) & (data['Age'] <=64), 'Age_band'] = 3
data.loc[data['Age'] >64, 'Age_band'] = 4
data.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Initial</th>
      <th>Age_band</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>Mrs</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>Miss</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>Mrs</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
data['Age_band'].value_counts().to_frame().style.background_gradient(cmap = 'summer_r')
```

<img src = 'https://user-images.githubusercontent.com/60168331/98236289-77ceb600-1fa6-11eb-9cc0-d35c4abf96da.png'>

```python
sns.factorplot('Age_band', 'Survived', data = data, col = 'Pclass')
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234645-14dc1f80-1fa4-11eb-97db-32a3482bded4.png'>

- Observations
    - 생존율은 객실등급과 무관하게 나이가 많을수록 떨어짐

<br>

#### Family_Size and Alone
- 새로운 Family_size와 Alone 변수를 만들것
- 두개의 변수는 Parch와 SibSP를 더해서 만듬
- 생존율이 탑승객 Family_size와 관련이 있는지 확인할 수 있도록 종합 데이터를 준다. 
- Alone 라는 것은 승객이 혼자인지 아닌지를 나타내는 것이다.


```python
data['Family_Size'] = 0
data['Family_Size'] = data['Parch'] + data['SibSp']

data['Alone'] = 0
data.loc[data.Family_Size == 0 , 'Alone'] = 1
```


```python
sns.factorplot('Family_Size', 'Survived', data = data)
plt.title('Family_Size vs Survived')

sns.factorplot('Alone', 'Survived', data = data)
plt.title('Alone vs Survived')

plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234648-160d4c80-1fa4-11eb-9f96-d1b46badbc08.png'>

<img src = 'https://user-images.githubusercontent.com/60168331/98234651-160d4c80-1fa4-11eb-89e5-481585a342b9.png'>


- Observation
    - Family_size가 0인것은 Alone이 1이다.
    - Family_size가 0과 4이상인것은 생존확률이 낮다
    - 다만 1~3은 생존확률이 높다


```python
sns.factorplot('Alone', 'Survived', data= data, hue = 'Sex', col = 'Pclass')
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234653-16a5e300-1fa4-11eb-8a68-d23e0a6252ab.png'>


- Observation
    - pclass3을 제외하면 성별과 객실등급과는 무관하게 혼자온것은 생존율이 낮다

<br>

#### Fare_Range
    - Fare도 연속적 수치이므로, 바꿔주도록 한다
    - pandas의 qcut을 이용하면 범위를 만들수 있음


```python
data['Fare_Range'] = pd.qcut(data['Fare'], 4)
data.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap = 'summer_r')
```

<img src = 'https://user-images.githubusercontent.com/60168331/98236438-b06e8f80-1fa6-11eb-8b5b-8acab8009e4f.png'>


- Age_band를 만든거 처럼 Fare_cat도 만듬


```python
data['Fare_cat'] = 0
data.loc[data['Fare'] <= 7.91, 'Fare_cat'] = 0
data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare_cat'] = 1
data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare_cat'] = 2
data.loc[(data['Fare'] > 31) & (data['Fare'] <= 513), 'Fare_cat'] = 3
```


```python
sns.factorplot('Fare_cat','Survived', data = data, hue = 'Sex')
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234655-16a5e300-1fa4-11eb-8d2f-74511eaa1bd6.png'>


- Observation
    - 요금이 높을수록 생존율이 높아짐

<br>

#### Converting String Values into Numeric
- Sex, Embarked, Initial 을 숫자로 변경


```python
data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
data['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
data['Initial'].replace(['Mr', 'Mrs', 'Miss', 'Master', 'Other', ], [
                        0, 1, 2, 3, 4], inplace=True)
```

<br>

#### Dropping UnNeeded Features
- Name, Age, Ticket, Fare, Cabin, Fare_Range, PassengerId


```python
data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'], axis = 1, inplace = True)
sns.heatmap(data.corr(), annot = True, cmap = 'RdYlGn', linewidths= 0.2, annot_kws={'size': 20})
fig = plt.gcf()
fig.set_size_inches(18,15)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234658-173e7980-1fa4-11eb-9615-9ccde8a67659.png'>

---
<br>


### Part3: Predictive Modeling
- 위의 EDA를 토대로 아래의 모델들에 적용시켜 보도록함
    - 1)Logistic Regression
    - 2)Support Vector Machines(Linear and radial)
    - 3)Random Forest
    - 4)K-Nearest Neighbours
    - 5)Naive Bayes
    - 6)Decision Tree
    - 7)Logistic Regression


```python
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
```


```python
train, test = train_test_split(
    data, test_size=0.3, random_state=0, stratify=data['Survived'])
train_X = train[train.columns[1:]]
train_Y = train[train.columns[:1]]
test_X = test[test.columns[1:]]
test_Y = test[test.columns[:1]]
X = data[data.columns[1:]]
Y = data['Survived']
```

<br>

#### Radial Support Vector Machines(rbf-SVM)


```python
model = svm.SVC(kernel= 'rbf', C = 1, gamma = 0.1)
model.fit(train_X, train_Y)
prediction1 = model.predict(test_X)
print('Accuracy for rbf SVM is', metrics.accuracy_score(prediction1, test_Y))
```

    Accuracy for rbf SVM is 0.835820895522388


<br>

#### Linear Support Vector Machine(linear-SVM)


```python
model = svm.SVC(kernel='linear', C = 0.1, gamma = 0.1)
model.fit(train_X, train_Y)
prediction2 = model.predict(test_X)
print('Accuracy for linear SVM is', metrics.accuracy_score(prediction2, test_Y))
```

    Accuracy for linear SVM is 0.8171641791044776


<br>

#### Logistic Regression


```python
model = LogisticRegression()
model.fit(train_X,train_Y)
prediction3=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction3,test_Y))
```

    The accuracy of the Logistic Regression is 0.8134328358208955


<br>

#### Decision Tree


```python
model=DecisionTreeClassifier()
model.fit(train_X,train_Y)
prediction4=model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction4,test_Y))
```

    The accuracy of the Decision Tree is 0.7985074626865671


<br>

#### K-Nearest Neighbours(KNN)


```python
model=KNeighborsClassifier() 
model.fit(train_X,train_Y)
prediction5=model.predict(test_X)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction5,test_Y))
```

    The accuracy of the KNN is 0.832089552238806


- KNN의 경우 N을 몇개로 두냐에 따라서 accuracy가 달라진다.


```python
a_index = list(range(1, 11))
a = pd.Series()
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in list(range(1,11)):
    model=KNeighborsClassifier(n_neighbors=i) 
    model.fit(train_X,train_Y)
    prediction=model.predict(test_X)
    a = a.append(pd.Series(metrics.accuracy_score(prediction, test_Y)))
plt.plot(a_index, a)
plt.xticks(x)
fig = plt.gcf()
fig.set_size_inches(12,6)
plt.show()
print('Accuracies for different values of n ar :', a.values, 'with the max value as', a.values.max())
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234664-186fa680-1fa4-11eb-9b78-ab9520219156.png'>


    Accuracies for different values of n ar : [0.75746269 0.79104478 0.80970149 0.80223881 0.83208955 0.81716418
     0.82835821 0.83208955 0.8358209  0.83208955] with the max value as 0.835820895522388


<br>

#### Gaussian Naive Bayes


```python
model = GaussianNB()
model.fit(train_X, train_Y)
prediction6 = model.predict(test_X)
print('The accuracy of the NaiveBayes is', metrics.accuracy_score(prediction6, test_Y))
```

    The accuracy of the NaiveBayes is 0.8134328358208955


<br>

#### Random Forests


```python
model = RandomForestClassifier(n_estimators= 100)
model.fit(train_X, train_Y)
prediction7 = model.predict(test_X)
print('The accuracy of the Random Forest is', metrics.accuracy_score(prediction7, test_Y))
```

    The accuracy of the Random Forest is 0.8134328358208955


- result
    - accuracy가 절대적인것은 아니다
    - test accuracy가 90%는 나와야한다
    - 지금은 90%가 나오니까 실패한것인가? 아니다 cross validation을 하면된다

<br>

#### Cross Validation
    - K-fold검증은 데이터를 5개로 나누고, 그중 1개만 검증에 사용하고 나머지 4개로 훈련한다
    - 위의 과정을 여러번 반복하는것
    - 과적합을 방지한다
    - 데이터가 모자를수도 있다(5개로 나누고 1개를 검증용으로 쓰기때문)


```python
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
```


```python
kfold = KFold(n_splits=10, random_state=22)
xyz = []
accuracy = []
std = []
classifiers = ['Linear Svm', 'Radial Svm', 'Logistic Regression',
               'KNN', 'Decision Tree', 'Naive Bayes', 'Random Forest']
models = [svm.SVC(kernel='linear'), svm.SVC(kernel='rbf'), LogisticRegression(
), KNeighborsClassifier(n_neighbors=9), DecisionTreeClassifier(), GaussianNB(), RandomForestClassifier(n_estimators=100)]

for i in models:
    model = i
    cv_result = cross_val_score(model, X, Y, cv = kfold, scoring= 'accuracy')
    cv_result = cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2 = pd.DataFrame({'CV Mean' : xyz, 'Std' : std}, index=classifiers)
new_models_dataframe2
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
      <th>CV Mean</th>
      <th>Std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Linear Svm</th>
      <td>0.793471</td>
      <td>0.047797</td>
    </tr>
    <tr>
      <th>Radial Svm</th>
      <td>0.828290</td>
      <td>0.034427</td>
    </tr>
    <tr>
      <th>Logistic Regression</th>
      <td>0.805843</td>
      <td>0.024061</td>
    </tr>
    <tr>
      <th>KNN</th>
      <td>0.813783</td>
      <td>0.041210</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>0.810362</td>
      <td>0.027879</td>
    </tr>
    <tr>
      <th>Naive Bayes</th>
      <td>0.801386</td>
      <td>0.028999</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>0.817066</td>
      <td>0.027947</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.subplots(figsize=(12,6))
box=pd.DataFrame(accuracy,index=[classifiers])
box.T.boxplot()
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234666-19083d00-1fa4-11eb-80a0-419da34e320a.png'>



```python
new_models_dataframe2['CV Mean'].plot.barh(width=0.8)
plt.title('Average CV Mean Accuracy')
fig=plt.gcf()
fig.set_size_inches(8,5)
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234667-19083d00-1fa4-11eb-894a-cf99ed064894.png'>


- classification accuracy는 불균형으로 인해 때때로 오도될 수 있다. 
- 우리는 confusion matrix의 도움으로 요약된 결과를 얻을 수 있는데, 이 매트릭스는 모델이 어디에서 잘못 되었는지 또는 모델이 잘못 예측한 클래스를 보여준다.

<br>

#### Confusion Matrix


```python
f,ax=plt.subplots(3,3,figsize=(12,10))
y_pred = cross_val_predict(svm.SVC(kernel='rbf'),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,0],annot=True,fmt='2.0f')
ax[0,0].set_title('Matrix for rbf-SVM')
y_pred = cross_val_predict(svm.SVC(kernel='linear'),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,1],annot=True,fmt='2.0f')
ax[0,1].set_title('Matrix for Linear-SVM')
y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,2],annot=True,fmt='2.0f')
ax[0,2].set_title('Matrix for KNN')
y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,0],annot=True,fmt='2.0f')
ax[1,0].set_title('Matrix for Random-Forests')
y_pred = cross_val_predict(LogisticRegression(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,1],annot=True,fmt='2.0f')
ax[1,1].set_title('Matrix for Logistic Regression')
y_pred = cross_val_predict(DecisionTreeClassifier(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,2],annot=True,fmt='2.0f')
ax[1,2].set_title('Matrix for Decision Tree')
y_pred = cross_val_predict(GaussianNB(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[2,0],annot=True,fmt='2.0f')
ax[2,0].set_title('Matrix for Naive Bayes')
plt.subplots_adjust(hspace=0.2,wspace=0.2)
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234669-19a0d380-1fa4-11eb-885a-8e39bb2a173d.png'>


- Interpreting Confusion Matrix
- 왼쪽 대각선은 각 클래스에 대해 올바른 예측의 수를 나타내고 오른쪽 대각선은 잘못된 예측의 수를 나타낸다. rbf-SVM에 대한 첫 번째 그림을 고려해 보십시오.

- 1)정확한 예측의 수는 491(죽은 경우) + 247(생존한 경우)이며, 평균 CV 정확도는 (491+247)/891 = 82.8%로 우리가 앞서 얻은 것이다.

- 2)Errors--> 58명의 사망자를 생존자로 잘못 분류하고 95명은 사망자로 생존했다. 그래서 그것은 죽은 사람이 살아남을 것이라고 예측함으로써 더 많은 실수를 저질렀다.

- 모든 행렬을 보면, 우리는 rbf-SVM이 죽은 승객을 정확하게 예측할 수 있는 더 높은 가능성을 가지고 있다고 말할 수 있지만, NaiveBayes는 생존한 승객을 정확하게 예측할 수 있는 더 높은 가능성을 가지고 있다.

<br>

#### Hyper-Parameters Tuning
- 기본 Parameter를 조정하거나 변경하여 더 나은 모델을 얻을 수 있다. 
- SVM 모델의 C와 감마처럼, 그리고 유사하게 다른 분류자에 대한 다른 매개변수들을 Hyper- Parameter라고 한다.
- 알고리즘의 학습 속도를 변경하고 더 나은 모델을 얻기 위해 튜닝할 수 있다. 이를 하이퍼 파라미터 튜닝(Hyper-Parameter Tuning이라고 한다.
- SVM과 RandomForests와 같은 두 가지 최고의 분류자에 대한 하이퍼 파라미터를 조정할 것이다.

<br>

#### SVM


```python
from sklearn.model_selection import GridSearchCV
C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['rbf','linear']
hyper={'kernel':kernel,'C':C,'gamma':gamma}
gd=GridSearchCV(estimator=svm.SVC(),param_grid=hyper,verbose=True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)
```

    Fitting 5 folds for each of 240 candidates, totalling 1200 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    0.8282593685267716
    SVC(C=0.4, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=0.3, kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)


    [Parallel(n_jobs=1)]: Done 1200 out of 1200 | elapsed:   13.9s finished


<br>

#### Random Forests


```python
n_estimators=range(100,1000,100)
hyper={'n_estimators':n_estimators}
gd=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=hyper,verbose=True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)
```

    Fitting 5 folds for each of 9 candidates, totalling 45 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done  45 out of  45 | elapsed:   30.5s finished


    0.819327098110602
    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=300,
                           n_jobs=None, oob_score=False, random_state=0, verbose=0,
                           warm_start=False)


- SVM은 0.828, Random Forest는 0.819의 정확도를 가진다, Random Forest는 n_estimators = 300으로 나왔다

<br>

#### Ensembling
- Ensembling은 모델의 정확성이나 성능을 높이는 좋은 방법이다. 간단히 말해서, 그것은 하나의 강력한 모델을 만들기 위한 다양한 단순한 모델들의 결합이다.
- Ensembling은 아래와 같은 방법으로 할수 있음

    - 1)Voting Classifier
    - 2)Bagging
    - 3)Boosting.

- Voting Classifier
    - 모든 서브모듈의 예측을 바탕으로 평균 예측 결과를 제공한다


```python
from sklearn.ensemble import VotingClassifier
ensemble_lin_rbf = VotingClassifier(estimators=[('KNN', KNeighborsClassifier(n_neighbors=10)),
                                               ('RBF', svm.SVC(probability=True,
                                                               kernel='rbf', C=0.5, gamma=0.1)),
                                               ('RFor', RandomForestClassifier(
                                                   n_estimators=500, random_state=0)),
                                               ('LR', LogisticRegression(C=0.05)),
                                               ('DT', DecisionTreeClassifier(
                                                   random_state=0)),
                                               ('NB', GaussianNB()),
                                               ('svm', svm.SVC(
                                                   kernel='linear', probability=True))
                                               ],
                                   voting='soft').fit(train_X, train_Y)

print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(test_X,test_Y))
cross=cross_val_score(ensemble_lin_rbf,X,Y, cv = 10,scoring = "accuracy")
print('The cross validated score is',cross.mean())
```

    The accuracy for ensembled model is: 0.8208955223880597
    The cross validated score is 0.8249188514357053


- Bagging
    - 데이터 집합의 작은 파티션에 유사한 분류자를 적용한 다음 모든 예측의 평균을 취함으로써 작동한다. 평균값 때문에, 분산이 감소한다. Voting Classifier와는 달리, Bagging은 유사한 분류자를 사용한다.

- Bagged KNN
    - Bagging은 분산성이 높은 모델과 함께 가장 잘 작동한다. 이에 대한 예로는 Decision Tree 또는 Random Forest가 있다. 우리는 n_neighbours의 작은 값을 가진 KNN을 n_neighbours의 작은 값으로 사용할 수 있다.


```python
from sklearn.ensemble import BaggingClassifier
model=BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),random_state=0,n_estimators=700)
model.fit(train_X,train_Y)
prediction=model.predict(test_X)
print('The accuracy for bagged KNN is:',metrics.accuracy_score(prediction,test_Y))
result=cross_val_score(model,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for bagged KNN is:',result.mean())
```

    The accuracy for bagged KNN is: 0.835820895522388
    The cross validated score for bagged KNN is: 0.8160424469413232


- Bagged DecisionTree


```python
model=BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=0,n_estimators=100)
model.fit(train_X,train_Y)
prediction=model.predict(test_X)
print('The accuracy for bagged Decision Tree is:',metrics.accuracy_score(prediction,test_Y))
result=cross_val_score(model,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for bagged Decision Tree is:',result.mean())
```

    The accuracy for bagged Decision Tree is: 0.8246268656716418
    The cross validated score for bagged Decision Tree is: 0.8227590511860174


-  Boosting
    - Boosting 분류자의 순차적 학습을 사용하는 앙상블 기법이다. 약한 모델의 단계적 고도화다.Boosting 다음과 같이 작동한다.
    - 모델은 먼저 완전한 데이터 집합에 대해 교육된다. 이제 그 모델은 몇몇 사례를 맞힐 것이고 몇몇은 틀릴 것이다. 이제 다음 반복에서 학습자는 잘못 예측된 사례에 더 초점을 맞추거나 더 비중 있게 다룰 것이다. 따라서 잘못된 사례를 정확하게 예측하려고 할 것이다. 이제 이 반복적인 과정은 계속되며, 정확도에 대한 한계에 도달할 때까지 모델에 새로운 분류기가 추가된다.

- AdaBoost(Adaptive Boosting)
    - 이 경우 학습자 또는 평가자가 약한 사람은 의사결정 나무다. 그러나 우리는 dafault base_estimator를 우리가 선택한 어떤 알고리즘으로도 바꿀 수 있다.


```python
from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)
result=cross_val_score(ada,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for AdaBoost is:',result.mean())
```

    The cross validated score for AdaBoost is: 0.8249188514357055


- Stochastic Gradient Boosting
    - Here too the weak learner is a Decision Tree.


```python
from sklearn.ensemble import GradientBoostingClassifier
grad=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)
result=cross_val_score(grad,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for Gradient Boosting is:',result.mean())
```

    The cross validated score for Gradient Boosting is: 0.8115230961298376


- XGBoost


```python
import xgboost as xg
xgboost=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
result=cross_val_score(xgboost,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for XGBoost is:',result.mean())
```

    The cross validated score for XGBoost is: 0.8160299625468165


<br>

#### Hyper-Parameter Tuning for AdaBoost


```python
n_estimators=list(range(100,1100,100))
learn_rate=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
hyper={'n_estimators':n_estimators,'learning_rate':learn_rate}
gd=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=hyper,verbose=True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)
```

    Fitting 5 folds for each of 120 candidates, totalling 600 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    0.8293892411022534
    AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.1,
                       n_estimators=100, random_state=None)


    [Parallel(n_jobs=1)]: Done 600 out of 600 | elapsed:  7.0min finished


<br>

#### Confusion Matrix for the Best Model


```python
ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.05)
result=cross_val_predict(ada,X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,result),cmap='winter',annot=True,fmt='2.0f')
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234670-19a0d380-1fa4-11eb-99ed-a605f2fee227.png'>


<br>

#### Feature Importance


```python
f,ax=plt.subplots(2,2,figsize=(15,12))
model=RandomForestClassifier(n_estimators=500,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,0])
ax[0,0].set_title('Feature Importance in Random Forests')
model=AdaBoostClassifier(n_estimators=200,learning_rate=0.05,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,1],color='#ddff11')
ax[0,1].set_title('Feature Importance in AdaBoost')
model=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,0],cmap='RdYlGn_r')
ax[1,0].set_title('Feature Importance in Gradient Boosting')
model=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,1],color='#FD0F00')
ax[1,1].set_title('Feature Importance in XgBoost')
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/98234671-1a396a00-1fa4-11eb-8d83-94080287ae7f.png'>


<br>

#### Result:
- 일반적으로 중요한 기능으로는 Initial,Par_cat,Pclass,Family_Size 등이 있다.

- Sex 특성은 전혀 중요성을 부여하지 않는 것 같은데, 앞서 살펴본 바와 같이 Sex와 Pclass의 결합이 매우 좋은 차별화 요소를 주고 있었다는 점에서 충격적이다. 성은 랜덤 포레스트에서만 중요하게 보인다.
- 그러나, 우리는 많은 분류자에서 맨 위에 있는 Initial을 볼 수 있다.우리는 이미 성별(Sex)과 이니셜(Initial)의 긍정적인 상관관계를 보았기 때문에 둘 다 성별을 가리킨다.

- P클래스 및 Fair_cat과 비슷하게 승객 및 Family_Size with Alone, Parch, SibSp의 상태를 말한다.
