---
title: Pymongo - MongoDB를 Python에서 사용해보기
author: HyunMin Kim
date: 2020-10-09 16:30:00 0000
categories: [DataBase, MongoDB]
tags: [MongoDB, Pymongo, Collection, Document]
---

## 1. Pymongo
---
### 1.1 Pymongo란?
- Python 에서 MongoDB 작업을 위한 패키지

<br>

### 1.2 설치
```python
pip install pymongo==2.8.1
```
- pip install pymongo==설치할 버전으로 설치함

<br>

## 2. Pymongo 실습
---
### 2.1 Import

```python
import pymongo
```

- Pymongo를 import 함

<br>

```python
import zigbang as zb
```

- zigbang의 매물을 크롤링하는 패키지를 import 함
- 위의 패키지는 따로 만든 패키지로, 실제로는 없음

<br>

```python
import pandas as pd
```

- 데이터프레임으로 생성을 하기 위해 판다스를 import함

<br>

### 2.2 데이터 베이스 서버 연결

```python
client = pymongo.MongoClient("mongodb://user:passwd@ip:27017/")
client
```
    MongoClient('ip주소', 27017)

- MongoDB의 서버와 연결하여 client 객체를 생성함
- 연결은 pymongo의 Mongoclient를 사용하여 ip와 port를 입력함

<br>

### 2.3 데이터 베이스와 컬렉션 생성

```python
zigbang = client.crawling.zigbang
zigbang
```
    Collection(Database(MongoClient('ip주소', 27017), 'crawling'), 'zigbang')

- crawling 이라는 데이터베이스에 zigbang이라는 컬렉션을 생성함
- client(위에서 생성한 객체).database.collection 으로 생성

<br>

### 2.4 Insert

```python
items = zb.oneroom("성수동")
len(items)
```
    9

- Zigbang패키지로 성수동의 원룸 매물을 크롤링함
- 크롤링된 정보는 Dictionary형태로 Key : Value 로 여러 정보들이 저장되어있음
- 총 9개의 매물이 크롤링되었음

<br>

```python
ids = zigbang.insert(items)
len(ids)
```
    91

- 위에서 크롤링한 데이터를 zigbang이라는 컬렉션에 insert함
- 이전에 크롤링된 데이터까지 하여 91개의 데이터가 저장되어 있음

<br>

### 2.5 데이터 프레임 저장

```python
sungsoo_df = pd.DataFrame(items)
columns = ["_floor", "address1", "rent", "size", "deposit", "options"]
sungsoo_df = sungsoo_df[columns]
sungsoo_df.tail(2)
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
      <th>_floor</th>
      <th>address1</th>
      <th>rent</th>
      <th>size</th>
      <th>deposit</th>
      <th>options</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>89</th>
      <td>2층</td>
      <td>서울시 성동구 성수동2가</td>
      <td>0</td>
      <td>16.0</td>
      <td>23000</td>
      <td>-</td>
    </tr>
    <tr>
      <th>90</th>
      <td>3층</td>
      <td>서울시 성동구 성수동1가</td>
      <td>60</td>
      <td>15.0</td>
      <td>3000</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
</div>

<br>

- 크롤링된 정보 중 필요한 정보들(층, 주소, 월세, 방 사이즈, 보증금, 옵션)만 sungsoo_df라는 데이터프레임으로 생성함

<br>

### 2.6 필요한 정보만 다시 Insert

```python
zigbang2 = client.crawling.zigbang2
```

- 위에서 정리한 sungsoo_df를 crawling 데이터베이스에 zigbang2라는 collection에 넣기 위해 collection을 재 생성함

<br>

```python
items = sungsoo_df.to_dict("records")
```

- df.to_dict('records')를 사용하여 데이터프레임을 리스트안에 딕셔너리로 만듬
- to_dict는 판다스의 기능으로 데이터프레임을 딕셔너리로 만들어주며, 아래의 옵션들을 가진다
    - dict(기본값) : dict like {column -> {index -> value}}
    - list : dict like {column -> [values]}
    - series : dict like {column -> Series(values)}
    - split : dict like {index -> [index], columns -> [columns], data -> [values]}
    - records : list like [{column -> value}, ... , {column -> value}]
    - index : dict like {index -> {column -> value}}

<br>

```python
items[0]
```
    {'_floor': '4층',
     'address1': '서울시 성동구 성수동1가',
     'rent': 50,
     'size': 8.0,
     'deposit': 1000,
     'options': '에어컨,냉장고,세탁기,인덕션,옷장,신발장,싱크대'}

- 데이터 1개 확인
- 위의 딕셔너리 형식이 리스트안에 생성되어 있음

<br>

```python
ids = zigbang2.insert(items)
```

- zigbang2 컬렉션에 itmes를 insert함

<br>

### 2.7 Find

```python
query = 질의
results = zigbang2.find(query)
```

- find를 하기 위해서 컬렉션에 query에 조회하는 질의(조건)을 넣어서 result 변수에 넣어줌 

<br>

```python
QUERY = {"rent": {"$lte": 50}}
results = zigbang2.find(QUERY)
pd.DataFrame(results).tail()
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
      <th>_floor</th>
      <th>_id</th>
      <th>address1</th>
      <th>deposit</th>
      <th>options</th>
      <th>rent</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>61</th>
      <td>5층</td>
      <td>5dbbdfeaa547631b003bbd04</td>
      <td>서울시 성동구 성수동2가</td>
      <td>26000</td>
      <td>-</td>
      <td>0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>62</th>
      <td>3층</td>
      <td>5dbbdfeaa547631b003bbd05</td>
      <td>서울시 성동구 성수동1가</td>
      <td>35000</td>
      <td>-</td>
      <td>0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>63</th>
      <td>2층</td>
      <td>5dbbdfeaa547631b003bbd08</td>
      <td>서울시 성동구 성수동1가</td>
      <td>18000</td>
      <td>-</td>
      <td>0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>64</th>
      <td>옥탑방</td>
      <td>5dbbdfeaa547631b003bbd0a</td>
      <td>서울시 성동구 성수동1가</td>
      <td>1000</td>
      <td>-</td>
      <td>50</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>65</th>
      <td>2층</td>
      <td>5dbbdfeaa547631b003bbd0b</td>
      <td>서울시 성동구 성수동2가</td>
      <td>23000</td>
      <td>-</td>
      <td>0</td>
      <td>16.0</td>
    </tr>
  </tbody>
</table>
</div>

<br>

- $lte를 사용하여 rent(월세)가 50이하인 매물만 find하는 Query를 생성
- collection.find(query)를 사용하여 조회함
- 조회한 데이터를 데이터프레임으로 생성 한 뒤 마지막 5개 매물만 조회


### 2.8 컬렉션 삭제

```python
client.crawling.drop_collection("zigbang")
```

- drop_collecntion(컬렉션)이라는 메서드를 사용해서 삭제함

<br>

### 2.9 데이터 베이스 삭제

```python
client.drop_database("crawling")
```

- drop_database(데이터베이스)이라는 메서드를 사용해서 삭제함
