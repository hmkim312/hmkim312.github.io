---
title: Good Books 데이터로 해보는 추천 시스템(Recommendations)
author: HyunMin Kim
date: 2020-10-26 10:00:00 0000
categories: [Data Science, Machine Learning]
tags: [Recommendations, Tfidf Vectorizer, linear Kernel]
---

## 1. Good Books
---
### 1.1 Good Books 데이터

<https://www.kaggle.com/zygmunt/goodbooks-10k>{:target="_blank"}

- ratings, books,tag, book_tags, to_read의 10k(10,000) 데이터

<br>

## 2. 추천 시스템 실습
---
### 2.1 Data load
#### 2.1.1 Books data


```python
import numpy as np
import pandas as pd

books = pd.read_csv('https://media.githubusercontent.com/media/hmkim312/datas/main/goodbooks-10k/books.csv', encoding='ISO-8859-1')
books.head()
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
      <th>book_id</th>
      <th>best_book_id</th>
      <th>work_id</th>
      <th>books_count</th>
      <th>isbn</th>
      <th>isbn13</th>
      <th>authors</th>
      <th>original_publication_year</th>
      <th>original_title</th>
      <th>...</th>
      <th>ratings_count</th>
      <th>work_ratings_count</th>
      <th>work_text_reviews_count</th>
      <th>ratings_1</th>
      <th>ratings_2</th>
      <th>ratings_3</th>
      <th>ratings_4</th>
      <th>ratings_5</th>
      <th>image_url</th>
      <th>small_image_url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2767052</td>
      <td>2767052</td>
      <td>2792775</td>
      <td>272</td>
      <td>439023483</td>
      <td>9.780439e+12</td>
      <td>Suzanne Collins</td>
      <td>2008.0</td>
      <td>The Hunger Games</td>
      <td>...</td>
      <td>4780653</td>
      <td>4942365</td>
      <td>155254</td>
      <td>66715</td>
      <td>127936</td>
      <td>560092</td>
      <td>1481305</td>
      <td>2706317</td>
      <td>https://images.gr-assets.com/books/1447303603m...</td>
      <td>https://images.gr-assets.com/books/1447303603s...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>4640799</td>
      <td>491</td>
      <td>439554934</td>
      <td>9.780440e+12</td>
      <td>J.K. Rowling, Mary GrandPrÃ©</td>
      <td>1997.0</td>
      <td>Harry Potter and the Philosopher's Stone</td>
      <td>...</td>
      <td>4602479</td>
      <td>4800065</td>
      <td>75867</td>
      <td>75504</td>
      <td>101676</td>
      <td>455024</td>
      <td>1156318</td>
      <td>3011543</td>
      <td>https://images.gr-assets.com/books/1474154022m...</td>
      <td>https://images.gr-assets.com/books/1474154022s...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>41865</td>
      <td>41865</td>
      <td>3212258</td>
      <td>226</td>
      <td>316015849</td>
      <td>9.780316e+12</td>
      <td>Stephenie Meyer</td>
      <td>2005.0</td>
      <td>Twilight</td>
      <td>...</td>
      <td>3866839</td>
      <td>3916824</td>
      <td>95009</td>
      <td>456191</td>
      <td>436802</td>
      <td>793319</td>
      <td>875073</td>
      <td>1355439</td>
      <td>https://images.gr-assets.com/books/1361039443m...</td>
      <td>https://images.gr-assets.com/books/1361039443s...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2657</td>
      <td>2657</td>
      <td>3275794</td>
      <td>487</td>
      <td>61120081</td>
      <td>9.780061e+12</td>
      <td>Harper Lee</td>
      <td>1960.0</td>
      <td>To Kill a Mockingbird</td>
      <td>...</td>
      <td>3198671</td>
      <td>3340896</td>
      <td>72586</td>
      <td>60427</td>
      <td>117415</td>
      <td>446835</td>
      <td>1001952</td>
      <td>1714267</td>
      <td>https://images.gr-assets.com/books/1361975680m...</td>
      <td>https://images.gr-assets.com/books/1361975680s...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>4671</td>
      <td>4671</td>
      <td>245494</td>
      <td>1356</td>
      <td>743273567</td>
      <td>9.780743e+12</td>
      <td>F. Scott Fitzgerald</td>
      <td>1925.0</td>
      <td>The Great Gatsby</td>
      <td>...</td>
      <td>2683664</td>
      <td>2773745</td>
      <td>51992</td>
      <td>86236</td>
      <td>197621</td>
      <td>606158</td>
      <td>936012</td>
      <td>947718</td>
      <td>https://images.gr-assets.com/books/1490528560m...</td>
      <td>https://images.gr-assets.com/books/1490528560s...</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



- Book에 대한 정보가 담긴 csv 파일
- 이번 데이터들은 encoding을 ISO-8859-1로 읽어야함
- rating 1 ~ 5의 의미는 별점 1점부터 5점의 갯수임

<br>

#### 2.1.2 Ratings Data


```python
ratings = pd.read_csv('https://media.githubusercontent.com/media/hmkim312/datas/main/goodbooks-10k/ratings.csv', encoding='ISO-8859-1')
ratings.head()
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
      <th>book_id</th>
      <th>user_id</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>314</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>439</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>588</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1169</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1185</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



- rating 데이터에는 Book_id와 User_id 그리고 해당 유저가 준 rating 점수가 있음

<br>

#### 2.1.3 Book tags Data load


```python
book_tags = pd.read_csv('https://media.githubusercontent.com/media/hmkim312/datas/main/goodbooks-10k/book_tags.csv', encoding='ISO-8859-1')
book_tags.head()
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
      <th>goodreads_book_id</th>
      <th>tag_id</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>30574</td>
      <td>167697</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>11305</td>
      <td>37174</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>11557</td>
      <td>34173</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>8717</td>
      <td>12986</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>33114</td>
      <td>12716</td>
    </tr>
  </tbody>
</table>
</div>



- Book의 id와 tag의 id가 있음

<br>

#### 2.1.4 Tags Data load


```python
tags = pd.read_csv('https://media.githubusercontent.com/media/hmkim312/datas/main/goodbooks-10k/tags.csv')
tags.tail()
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
      <th>tag_id</th>
      <th>tag_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34247</th>
      <td>34247</td>
      <td>Ｃhildrens</td>
    </tr>
    <tr>
      <th>34248</th>
      <td>34248</td>
      <td>Ｆａｖｏｒｉｔｅｓ</td>
    </tr>
    <tr>
      <th>34249</th>
      <td>34249</td>
      <td>Ｍａｎｇａ</td>
    </tr>
    <tr>
      <th>34250</th>
      <td>34250</td>
      <td>ＳＥＲＩＥＳ</td>
    </tr>
    <tr>
      <th>34251</th>
      <td>34251</td>
      <td>ｆａｖｏｕｒｉｔｅｓ</td>
    </tr>
  </tbody>
</table>
</div>



- Tag의 id와 해당 tag와 연결되는 name이 있음

<br>

#### 2.1.5 Read Data load


```python
to_read = pd.read_csv('https://media.githubusercontent.com/media/hmkim312/datas/main/goodbooks-10k/to_read.csv')
to_read.head()
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
      <th>user_id</th>
      <th>book_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>112</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>235</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>533</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1198</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1874</td>
    </tr>
  </tbody>
</table>
</div>



- 유저가 어떤 책을 읽었는지에 대한 id가 적혀있음

<br>

### 2.2 Tag Data 전처리


```python
tags_join_Df = pd.merge(book_tags, tags, left_on='tag_id', right_on='tag_id', how = 'inner')
tags_join_Df.head()
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
      <th>goodreads_book_id</th>
      <th>tag_id</th>
      <th>count</th>
      <th>tag_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>30574</td>
      <td>167697</td>
      <td>to-read</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>30574</td>
      <td>24549</td>
      <td>to-read</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>30574</td>
      <td>496107</td>
      <td>to-read</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>30574</td>
      <td>11909</td>
      <td>to-read</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>30574</td>
      <td>298</td>
      <td>to-read</td>
    </tr>
  </tbody>
</table>
</div>



- Tagid와 tag_name을 books id가 있는 데이터 프레임과 merge함

<br>

### 2.3 Authors로 Tfidf


```python
books['authors'][:3]
```




    0                 Suzanne Collins
    1    J.K. Rowling, Mary GrandPrÃ©
    2                 Stephenie Meyer
    Name: authors, dtype: object



- books 데이터에는 작가명 컬럼이 있음

<br>


```python
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df = 0, stop_words = 'english')
tfidf_matrix = tf.fit_transform(books['authors'])
tfidf_matrix
```




    <10000x14742 sparse matrix of type '<class 'numpy.float64'>'
    	with 43235 stored elements in Compressed Sparse Row format>



- Books에 있는 작가명으로 Tfidf를 수행함

<br>

### 2.4 유사도 측정


```python
from sklearn.metrics.pairwise import linear_kernel

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim
```




    array([[1., 0., 0., ..., 0., 0., 0.],
           [0., 1., 0., ..., 0., 0., 0.],
           [0., 0., 1., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 1., 0., 0.],
           [0., 0., 0., ..., 0., 1., 0.],
           [0., 0., 0., ..., 0., 0., 1.]])



- 사이킷런의 linear_kernel을 사용하여 작가명으로 만든 Tfidf매트릭스를 유사도 행렬로 생성

<br>

### 2.5 Hobbit과 유사한 책은?


```python
title = books['title']
indices = pd.Series(books.index, index=books['title'])
indices['The Hobbit']
```




    6



- Hobbit의 index는 6번이다
- 6번 행을 불러와서 비슷한 책을 찾게 해보자

<br>


```python
cosine_sim[indices['The Hobbit']]
```




    array([0., 0., 0., ..., 0., 0., 0.])



- 유사도 행렬에서 hobbit의 인덱스의 행을 불러옴

<br>


```python
cosine_sim[indices['The Hobbit']].shape
```




    (10000,)



- 총 1만개의 책 데이터가 있음

<br>


```python
list(enumerate(cosine_sim[indices['The Hobbit']]))[:3]
```




    [(0, 0.0), (1, 0.0), (2, 0.0)]



- 유사도 행렬에서 The Hobbit의 인덱스만 가져오고, 해당 컬럼(다른책 책 인덱스)와 코사인 유사도 점수를 enumerate를 사용하여 튜플형식으로 만들고, 해당 데이터를 list에 넣는다

<br>

### 2.6 가장 유사한 책의 Index


```python
sim_scores = list(enumerate(cosine_sim[indices['The Hobbit']]))
sim_scores = sorted(sim_scores, key = lambda x : x[1], reverse= True)
sim_scores[:3]
```




    [(6, 1.0), (18, 1.0), (154, 1.0)]



- 호빗과 가장 유사한 책의 인덱스(여기서는 열)와 코사인 점수를 정렬하여 출력함
- 완전 똑같은 1점도 보인다. 18번, 154번
- 참고로 맨 앞에 (6, 1.0)은 본인 자신임

<br>


```python
print(f'Index 6번의 책 이름 :', books['title'][6])
print(f'Index 18번의 책 이름 :', books['title'][18])
print(f'Index 154번의 책 이름 :', books['title'][154])
```

    Index 6번의 책 이름 : The Hobbit
    Index 18번의 책 이름 : The Fellowship of the Ring (The Lord of the Rings, #1)
    Index 154번의 책 이름 : The Two Towers (The Lord of the Rings, #2)


- 호빗과 비슷한 책은 반지의 제왕 시리즈가 나옴

<br>

### 2.7 작가로 본 유사 책 검색


```python
sim_scores = sim_scores[1:11]
book_indices = [i[0] for i in sim_scores]
title.iloc[book_indices]
```




    18      The Fellowship of the Ring (The Lord of the Ri...
    154            The Two Towers (The Lord of the Rings, #2)
    160     The Return of the King (The Lord of the Rings,...
    188     The Lord of the Rings (The Lord of the Rings, ...
    963     J.R.R. Tolkien 4-Book Boxed Set: The Hobbit an...
    4975        Unfinished Tales of NÃºmenor and Middle-Earth
    2308                               The Children of HÃºrin
    610              The Silmarillion (Middle-Earth Universe)
    8271                   The Complete Guide to Middle-Earth
    1128     The History of the Hobbit, Part One: Mr. Baggins
    Name: title, dtype: object



- 그 외의 다른 책들도 대부분 Hobbit이긴 하나, 아마 작가가 동일인일 가능성이 높다.
- 사실 생각해 보면 작가이름으로만 Tfidf를 했기 때문에, 작가 이름이 같다면 모두 동일한 점수(1)로 나올것이다

<br>

### 2.8 Tag 추가


```python
books_with_tags = pd.merge(books, tags_join_Df, left_on= 'book_id', right_on='goodreads_book_id', how = 'inner')
books_with_tags.head()
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
      <th>book_id</th>
      <th>best_book_id</th>
      <th>work_id</th>
      <th>books_count</th>
      <th>isbn</th>
      <th>isbn13</th>
      <th>authors</th>
      <th>original_publication_year</th>
      <th>original_title</th>
      <th>...</th>
      <th>ratings_2</th>
      <th>ratings_3</th>
      <th>ratings_4</th>
      <th>ratings_5</th>
      <th>image_url</th>
      <th>small_image_url</th>
      <th>goodreads_book_id</th>
      <th>tag_id</th>
      <th>count</th>
      <th>tag_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2767052</td>
      <td>2767052</td>
      <td>2792775</td>
      <td>272</td>
      <td>439023483</td>
      <td>9.780439e+12</td>
      <td>Suzanne Collins</td>
      <td>2008.0</td>
      <td>The Hunger Games</td>
      <td>...</td>
      <td>127936</td>
      <td>560092</td>
      <td>1481305</td>
      <td>2706317</td>
      <td>https://images.gr-assets.com/books/1447303603m...</td>
      <td>https://images.gr-assets.com/books/1447303603s...</td>
      <td>2767052</td>
      <td>30574</td>
      <td>11314</td>
      <td>to-read</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2767052</td>
      <td>2767052</td>
      <td>2792775</td>
      <td>272</td>
      <td>439023483</td>
      <td>9.780439e+12</td>
      <td>Suzanne Collins</td>
      <td>2008.0</td>
      <td>The Hunger Games</td>
      <td>...</td>
      <td>127936</td>
      <td>560092</td>
      <td>1481305</td>
      <td>2706317</td>
      <td>https://images.gr-assets.com/books/1447303603m...</td>
      <td>https://images.gr-assets.com/books/1447303603s...</td>
      <td>2767052</td>
      <td>11305</td>
      <td>10836</td>
      <td>fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2767052</td>
      <td>2767052</td>
      <td>2792775</td>
      <td>272</td>
      <td>439023483</td>
      <td>9.780439e+12</td>
      <td>Suzanne Collins</td>
      <td>2008.0</td>
      <td>The Hunger Games</td>
      <td>...</td>
      <td>127936</td>
      <td>560092</td>
      <td>1481305</td>
      <td>2706317</td>
      <td>https://images.gr-assets.com/books/1447303603m...</td>
      <td>https://images.gr-assets.com/books/1447303603s...</td>
      <td>2767052</td>
      <td>11557</td>
      <td>50755</td>
      <td>favorites</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2767052</td>
      <td>2767052</td>
      <td>2792775</td>
      <td>272</td>
      <td>439023483</td>
      <td>9.780439e+12</td>
      <td>Suzanne Collins</td>
      <td>2008.0</td>
      <td>The Hunger Games</td>
      <td>...</td>
      <td>127936</td>
      <td>560092</td>
      <td>1481305</td>
      <td>2706317</td>
      <td>https://images.gr-assets.com/books/1447303603m...</td>
      <td>https://images.gr-assets.com/books/1447303603s...</td>
      <td>2767052</td>
      <td>8717</td>
      <td>35418</td>
      <td>currently-reading</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2767052</td>
      <td>2767052</td>
      <td>2792775</td>
      <td>272</td>
      <td>439023483</td>
      <td>9.780439e+12</td>
      <td>Suzanne Collins</td>
      <td>2008.0</td>
      <td>The Hunger Games</td>
      <td>...</td>
      <td>127936</td>
      <td>560092</td>
      <td>1481305</td>
      <td>2706317</td>
      <td>https://images.gr-assets.com/books/1447303603m...</td>
      <td>https://images.gr-assets.com/books/1447303603s...</td>
      <td>2767052</td>
      <td>33114</td>
      <td>25968</td>
      <td>young-adult</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



- Books 데이터 프레임에, 앞에서 만든 tagid와 tag name을 merge함

<br>

### 2.9 Tag를 Tfidf


```python
tf_tag = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df = 0, stop_words='english')
tfidf_matrix_tag = tf_tag.fit_transform(books_with_tags['tag_name'].head(10000))
cosine_sim_tag = linear_kernel(tfidf_matrix_tag, tfidf_matrix_tag)
```

- 앞에선 작가 이름으로 Tfidf를 했고, 이번엔 Tag로 해본다

<br>

### 2.10 추천책을 반환하는 함수


```python
title_tag = books['title']
indices_tag = pd.Series(books.index, index=books['title'])


def tags_recommendations(title):
    idx = indices_tag[title]
    sim_scores = list(enumerate(cosine_sim_tag[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    book_indices = [i[0] for i in sim_scores]
    return title_tag.iloc[book_indices]
```

- 이번에는 책의 제목을 넣으면 추천책을 반환하는 함수를 작성
- sim_scores = sim_scores[1:11]은 총 10개를 가리키며, 1부터 한것은 0번은 입력한 책 제목 자신이 나오기 떄문임

<br>

### 2.11 Tag로 찾아본 Hobbits와 유사책


```python
tags_recommendations('The Hobbit').head(20)
```




    16             Catching Fire (The Hunger Games, #2)
    31                                  Of Mice and Men
    107    Confessions of a Shopaholic (Shopaholic, #1)
    125                       Dune (Dune Chronicles #1)
    149                                    The Red Tent
    206          One for the Money (Stephanie Plum, #1)
    214                                Ready Player One
    231             The Gunslinger (The Dark Tower, #1)
    253          Shiver (The Wolves of Mercy Falls, #1)
    313                         Inkheart (Inkworld, #1)
    Name: title, dtype: object



- 헝거게임, 듄 등 호빗과 비슷한 판타지 장르가 나오는듯 싶다.

<br>

### 2.12 Book id에 tag name을 한번에 붙이기


```python
temp_df = books_with_tags.groupby('book_id')['tag_name'].apply(' '.join).reset_index()
temp_df.head()
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
      <th>book_id</th>
      <th>tag_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>to-read fantasy favorites currently-reading yo...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>to-read fantasy favorites currently-reading yo...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>to-read fantasy favorites currently-reading yo...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>to-read fantasy favorites currently-reading yo...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>to-read fantasy young-adult fiction harry-pott...</td>
    </tr>
  </tbody>
</table>
</div>



- Book Id에 있는 모든 tag_name들을 한번에 모아놓음

<br>

### 2.13 Boos에 Merge


```python
books = pd.merge(books, temp_df, on = 'book_id', how = 'inner')
books.head()
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
      <th>book_id</th>
      <th>best_book_id</th>
      <th>work_id</th>
      <th>books_count</th>
      <th>isbn</th>
      <th>isbn13</th>
      <th>authors</th>
      <th>original_publication_year</th>
      <th>original_title</th>
      <th>...</th>
      <th>work_ratings_count</th>
      <th>work_text_reviews_count</th>
      <th>ratings_1</th>
      <th>ratings_2</th>
      <th>ratings_3</th>
      <th>ratings_4</th>
      <th>ratings_5</th>
      <th>image_url</th>
      <th>small_image_url</th>
      <th>tag_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2767052</td>
      <td>2767052</td>
      <td>2792775</td>
      <td>272</td>
      <td>439023483</td>
      <td>9.780439e+12</td>
      <td>Suzanne Collins</td>
      <td>2008.0</td>
      <td>The Hunger Games</td>
      <td>...</td>
      <td>4942365</td>
      <td>155254</td>
      <td>66715</td>
      <td>127936</td>
      <td>560092</td>
      <td>1481305</td>
      <td>2706317</td>
      <td>https://images.gr-assets.com/books/1447303603m...</td>
      <td>https://images.gr-assets.com/books/1447303603s...</td>
      <td>to-read fantasy favorites currently-reading yo...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>4640799</td>
      <td>491</td>
      <td>439554934</td>
      <td>9.780440e+12</td>
      <td>J.K. Rowling, Mary GrandPrÃ©</td>
      <td>1997.0</td>
      <td>Harry Potter and the Philosopher's Stone</td>
      <td>...</td>
      <td>4800065</td>
      <td>75867</td>
      <td>75504</td>
      <td>101676</td>
      <td>455024</td>
      <td>1156318</td>
      <td>3011543</td>
      <td>https://images.gr-assets.com/books/1474154022m...</td>
      <td>https://images.gr-assets.com/books/1474154022s...</td>
      <td>to-read fantasy favorites currently-reading yo...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>41865</td>
      <td>41865</td>
      <td>3212258</td>
      <td>226</td>
      <td>316015849</td>
      <td>9.780316e+12</td>
      <td>Stephenie Meyer</td>
      <td>2005.0</td>
      <td>Twilight</td>
      <td>...</td>
      <td>3916824</td>
      <td>95009</td>
      <td>456191</td>
      <td>436802</td>
      <td>793319</td>
      <td>875073</td>
      <td>1355439</td>
      <td>https://images.gr-assets.com/books/1361039443m...</td>
      <td>https://images.gr-assets.com/books/1361039443s...</td>
      <td>to-read fantasy favorites currently-reading yo...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2657</td>
      <td>2657</td>
      <td>3275794</td>
      <td>487</td>
      <td>61120081</td>
      <td>9.780061e+12</td>
      <td>Harper Lee</td>
      <td>1960.0</td>
      <td>To Kill a Mockingbird</td>
      <td>...</td>
      <td>3340896</td>
      <td>72586</td>
      <td>60427</td>
      <td>117415</td>
      <td>446835</td>
      <td>1001952</td>
      <td>1714267</td>
      <td>https://images.gr-assets.com/books/1361975680m...</td>
      <td>https://images.gr-assets.com/books/1361975680s...</td>
      <td>to-read favorites currently-reading young-adul...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>4671</td>
      <td>4671</td>
      <td>245494</td>
      <td>1356</td>
      <td>743273567</td>
      <td>9.780743e+12</td>
      <td>F. Scott Fitzgerald</td>
      <td>1925.0</td>
      <td>The Great Gatsby</td>
      <td>...</td>
      <td>2773745</td>
      <td>51992</td>
      <td>86236</td>
      <td>197621</td>
      <td>606158</td>
      <td>936012</td>
      <td>947718</td>
      <td>https://images.gr-assets.com/books/1490528560m...</td>
      <td>https://images.gr-assets.com/books/1490528560s...</td>
      <td>to-read favorites currently-reading young-adul...</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



- 이번에는 tag name이 하나의 컬럼에 여러개가 들어있음

<br>

### 2.14 작가와 Tag name을 합침


```python
books['corpus'] = (pd.Series(books[['authors', 'tag_name']]
                            .fillna('')
                            .values.tolist()
                           ).str.join(' '))
books['corpus'][:3]
```




    0    Suzanne Collins to-read fantasy favorites curr...
    1    J.K. Rowling, Mary GrandPrÃ© to-read fantasy f...
    2    Stephenie Meyer to-read fantasy favorites curr...
    Name: corpus, dtype: object



- corpus라는 컬럼에 저자와 태그가 한번에 모두 있음

<br>

### 2.15 Tfidf 실행


```python
tf_corpus = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=0, stop_words='english')
tfidf_matrix_corpus = tf_corpus.fit_transform(books['corpus'])
cosine_sim_corpus = linear_kernel(tfidf_matrix_corpus, tfidf_matrix_corpus)
titles = books['title']
indices = pd.Series(books.index, index=books['title'])
```

- 작가와 Tag name을 합친것을 Tfidf를 실행함

<br>

### 2.16 추천 함수 작성


```python
def corpus_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim_corpus[idx]))
    sim_scores = sorted(sim_scores, key = lambda x : x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    book_indices = [i[0] for i in sim_scores]
    return titles.iloc[book_indices]
```

<br>

### 2.17 비슷한 책은?


```python
corpus_recommendations('The Hobbit')
```




    188     The Lord of the Rings (The Lord of the Rings, ...
    154            The Two Towers (The Lord of the Rings, #2)
    160     The Return of the King (The Lord of the Rings,...
    18      The Fellowship of the Ring (The Lord of the Ri...
    610              The Silmarillion (Middle-Earth Universe)
    4975        Unfinished Tales of NÃºmenor and Middle-Earth
    2308                               The Children of HÃºrin
    963     J.R.R. Tolkien 4-Book Boxed Set: The Hobbit an...
    465                             The Hobbit: Graphic Novel
    8271                   The Complete Guide to Middle-Earth
    Name: title, dtype: object



- The Hobbit과 비슷한 책은 이제 잘 나오는듯 하다.

<br>


```python
corpus_recommendations('Twilight (Twilight, #1)')
```




    51                                 Eclipse (Twilight, #3)
    48                                New Moon (Twilight, #2)
    991                    The Twilight Saga (Twilight, #1-4)
    833                         Midnight Sun (Twilight, #1.5)
    731     The Short Second Life of Bree Tanner: An Eclip...
    1618    The Twilight Saga Complete Collection  (Twilig...
    4087    The Twilight Saga: The Official Illustrated Gu...
    2020             The Twilight Collection (Twilight, #1-3)
    72                                The Host (The Host, #1)
    219     Twilight: The Complete Illustrated Movie Compa...
    Name: title, dtype: object



- 트와일라잇과 비슷한 책들

<br>


```python
corpus_recommendations('Harry Potter and the Prisoner of Azkaban (Harry Potter, #3)')
```




    1       Harry Potter and the Sorcerer's Stone (Harry P...
    26      Harry Potter and the Half-Blood Prince (Harry ...
    22      Harry Potter and the Chamber of Secrets (Harry...
    24      Harry Potter and the Deathly Hallows (Harry Po...
    23      Harry Potter and the Goblet of Fire (Harry Pot...
    20      Harry Potter and the Order of the Phoenix (Har...
    3752         Harry Potter Collection (Harry Potter, #1-6)
    398                          The Tales of Beedle the Bard
    1285                           Quidditch Through the Ages
    421              Harry Potter Boxset (Harry Potter, #1-7)
    Name: title, dtype: object



- 해리포터와 비슷한 책

<br>


```python
corpus_recommendations('Romeo and Juliet')
```




    352                      Othello
    769                Julius Caesar
    124                       Hamlet
    153                      Macbeth
    247    A Midsummer Night's Dream
    838       The Merchant of Venice
    854                Twelfth Night
    529       Much Ado About Nothing
    713                    King Lear
    772      The Taming of the Shrew
    Name: title, dtype: object



- 로미오와 줄리엣과 비슷한 책

<br>

## 3. 요약
---
### 3.1 요약

- 책 데이터로 해본 추천 시스템, Tfidf를 사용하였고, 사실 작가나 태그만 사용한다면 같은 작가, 같은 태그의 책들만 추천을 해줬을것이다.
- 하지만 하나의 컬럼에 모아서 Tfidf를 하였을땐 조금 다른 결과가 나왔으나, 이렇게 하는것이 맞는지, 혹은 더 다른 방법은 없는지 싶다
- 추천 시스템은 어려운듯 하다
