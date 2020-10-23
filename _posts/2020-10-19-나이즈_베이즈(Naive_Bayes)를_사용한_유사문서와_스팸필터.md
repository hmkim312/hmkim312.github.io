---
title: 나이즈 베이즈(Naive Bayes)를 사용한 유사문서와 스팸필터
author: HyunMin Kim
date: 2020-10-19 11:10:00 0000
categories: [Data Science, Machine Learning]
tags: [Fetch 20newsgroups, Tfidf Transformer, Count Vectorize, Pipeline, Sparse Matrix, Classification Report, Multinomial Naive Bayes, Confusion Matrix, Spam, Compression]
---

## 1. 나이브 베이즈를 활용한 유사 문서 검색
---
### 1.1 뉴스 문서 Load


```python
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=13)
twenty_train.keys()
```




    dict_keys(['data', 'filenames', 'target_names', 'target', 'DESCR'])



- Sklearn에는 20개의 카테고리의 뉴스 문서를 fetch_20newsgroups 메서드로 제공함
- 제공하는 뉴스 중 총 'alt.atheism(무신론)', 'soc.religion.christian(기독교)', 'comp.graphics(컴퓨터)', 'sci.med(의약)' 4개의 주제를 불러옴
- data, filenames, target_names, target, DESCR로 데이터는 이루어져 있음

<br>

### 1.2 데이터셋의 설명


```python
print(twenty_train['DESCR'])
```

    .. _20newsgroups_dataset:
    
    The 20 newsgroups text dataset
    ------------------------------
    
    The 20 newsgroups dataset comprises around 18000 newsgroups posts on
    20 topics split in two subsets: one for training (or development)
    and the other one for testing (or for performance evaluation). The split
    between the train and test set is based upon a messages posted before
    and after a specific date.
    ...
    

- 해당 데이터의 설명과 사용법 등이 저장되어있다.

<br>

### 1.3 타겟 이름과 데이터의 갯수


```python
print(len(twenty_train.data))
twenty_train.target_names
```

    2257
    ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']



- 아까 지정한 4개의 카테고리가 target name이며 총 2257개의 데이터(뉴스문서)가 있음

<br>

### 1.4 실제 데이터 내용


```python
print(twenty_train['data'][0])
```

    From: geb@cs.pitt.edu (Gordon Banks)
    Subject: Re: Update (Help!) [was "What is This [Is it Lyme's?]"]
    Article-I.D.: pitt.19436
    Reply-To: geb@cs.pitt.edu (Gordon Banks)
    Organization: Univ. of Pittsburgh Computer Science
    Lines: 42
    
    In article <1993Mar29.181958.3224@equator.com> jod@equator.com (John Setel O'Donnell) writes:
    >
    >I shouldn't have to be posting here.  Physicians should know the Lyme
    >literature beyond Steere & co's denial merry-go-round.  Patients
    >should get correctly diagnosed and treated.
    >

    ...
    -- 
    ----------------------------------------------------------------------------
    Gordon Banks  N3JXP      | "Skepticism is the chastity of the intellect, and
    geb@cadre.dsl.pitt.edu   |  it is shameful to surrender it too soon." 
    ----------------------------------------------------------------------------
    


- 첫번째 뉴스데이터를 봐보았으며, 의약 분야에 대한 신문 기사이다

<br>

### 1.5 타겟 확인


```python
print(twenty_train.target_names[twenty_train['target'][0]])
```

    sci.med


- 해당 뉴스 기사의 타겟이름은 sci.med 이다

<br>


```python
print(twenty_train.target_names)
twenty_train.target[:10]
```

    ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
    array([2, 1, 1, 3, 2, 3, 0, 3, 2, 0])



- 타겟이름의 순서대로 0~3까지 이다.

<br>

### 1.6 Count Vectorize


```python
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape
```




    (2257, 35788)




```python
X_train_counts
```




    <2257x35788 sparse matrix of type '<class 'numpy.int64'>'
    	with 365886 stored elements in Compressed Sparse Row format>



- 전체 2257개의 뉴스 데이터를 카운트 벡터라이즈한 결과 총 35788개의 단어가 생성되었음
- 해당 행렬은 2257 x 35788의 거대한 행렬임

<br>

### 1.7 거대한 행렬의 처리


```python
print(len(X_train_counts.toarray()[0]))
X_train_counts.toarray()[0]
```

    35788
    array([0, 0, 0, ..., 0, 0, 0])



- 첫번째 행을 보았으나, 대부분 0으로 이루어져있고, 가끔 1이나 그외의 값으로 입력이 되어있을것
- 행은 데이터, 열은 35788개의 단어 이기 때문임

<br>

### 1.8 희소행렬(Sparse matrix)과 행렬 압축(Compression)

<img src="https://user-images.githubusercontent.com/60168331/96452430-f829b300-1253-11eb-97b5-d48f2593ba52.png">

- 희소행렬 : 대부분의 값이 0으로 이루어져있는 행렬
- 행렬압축 : 행 열 값으로 이루어져, 희소행렬을 압축해놓은 행렬

<br>

### 1.9 값이 0이 아닌것


```python
import numpy as np

print('전체 데이터의 합 : ', np.sum(X_train_counts.toarray()[0]))
print('0이 아닌 데이터의 갯수 : ', len(X_train_counts.toarray()[0][X_train_counts.toarray()[0] != 0]))
```

    전체 데이터의 합 :  320
    0이 아닌 데이터의 갯수 :  193


- 총 35788의 열 중에서 언급된 단어 데이터는 320개
- 그 중 중복을 제거하고 0이 아닌 데이터의 행 갯수는 193개 임
- 총 35788개 중에 193개를 제외하고는 모두 0이라는 이야기 = 희소행렬

<br>

### 1.10 Tf-idf 적용


```python
from sklearn.feature_extraction.text import TfidfTransformer

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape
```




    (2257, 35788)



- Tfidf를 적용하여도 35788개의 열이 나온다.
- 결국 희소행렬임

<br>

### 1.11 Multinomial Naive Bayes 적용


```python
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tf, twenty_train.target)
```

- Multinomial Naive Bayes를 Tf-idf를 적용시킨 데이터와 target을 넣고 학습 시켜줌
- target = y 라고 생각해도 됨

<br>

### 1.12 간단 테스트


```python
docs_new =['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, categories in zip(docs_new, predicted):
    print(f'{doc} => {twenty_train.target_names[categories]}')
```

    God is love => soc.religion.christian
    OpenGL on the GPU is fast => comp.graphics


- 간단하게 God is love, OpenGL on the GPU is fast 라는 문장을 카운트 벡터라이즈 한뒤 TF-idf로 거리를 구하여 MultinolialNB을 적용하여 예측하였다.
- God is love는 soc.religion.christian으로 예측되고, OpenGL on the GPU is fast는 comp.grahics로 예측하는 것으로 보아 괜찮은듯 하다

<br>

### 1.13 PipeLine 생성


```python
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
```

- 문장이 입력되면 Pipeline을 통해 카운트벡터라이즈 -> Tf-idf -> MultinomialNB를 거치는 Pipeline을 생성

<br>

### 1.14 학습 후 Test


```python
text_clf.fit(twenty_train.data, twenty_train.target)

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=13)

docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)
```




    0.8348868175765646



- 위에서 생성한 pipelien을 통과하여 train data를 학습시키고, 다시 같은 카테고리로 test 데이터를 불러와 예측해봄
- 예측된 결과와 실제 타겟은 약 0.83의 Accuracty를 가짐

<br>

### 1.15 Classification Report


```python
from sklearn.metrics import classification_report

print(classification_report(twenty_test.target, predicted, target_names= twenty_test.target_names))
```

                            precision    recall  f1-score   support
    
               alt.atheism       0.97      0.60      0.74       319
             comp.graphics       0.96      0.89      0.92       389
                   sci.med       0.97      0.81      0.88       396
    soc.religion.christian       0.65      0.99      0.78       398
    
                  accuracy                           0.83      1502
                 macro avg       0.89      0.82      0.83      1502
              weighted avg       0.88      0.83      0.84      1502
    


- atheism과 christain에서 상대적으로 f1-score가 낮은것으로 보아, 해당 뉴스들이 비슷한 단어들을 많이 쓰는듯 하다(종교적인 이야기이니 당연한걸지도..)

<br>

### 1.16 Confusion Matrix


```python
from sklearn.metrics import confusion_matrix

confusion_matrix(twenty_test.target, predicted)
```




    array([[192,   2,   6, 119],
           [  2, 347,   4,  36],
           [  2,  11, 322,  61],
           [  2,   2,   1, 393]])



- Confusion Matrix는 아래 와 같이 본다.
- atheism을 atheism으로 예측한것은 192개, graphics는 2개, med는 6개 christin은 119개
- graphics는 atheism으로 예측한것은 2개, graphics는 347개, med는 4개 christin은 36개

<br>

## 2. SMS Spam Collection Data
---
### 2.1 Data

- kaggle에 있는 데이터로 문자를 받고 스팸인지 아닌지를 구분하는 데이터 셋
- 참고로 스팸의 반대말은 햄이다..

<br>

### 2.2 Data load


```python
import pandas as pd

messages = pd.read_csv('https://raw.githubusercontent.com/hmkim312/datas/main/spam/spam.csv', encoding='latin-1')
messages.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1, inplace=True)
messages = messages.rename(columns={'v1' : 'class', 'v2' : 'text'})
messages.head()
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
      <th>class</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
    </tr>
  </tbody>
</table>
</div>



- 해당 데이터를 불러왔고, 데이터는 깃헙에 있다
- 보면 class 는 target이고, text는 내용이다
- 필요없는 3개의 컬럼은 삭제하였다

<br>

### 2.3 Data 통계


```python
messages.groupby('class').describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">text</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
    <tr>
      <th>class</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ham</th>
      <td>4825</td>
      <td>4516</td>
      <td>Sorry, I'll call later</td>
      <td>30</td>
    </tr>
    <tr>
      <th>spam</th>
      <td>747</td>
      <td>653</td>
      <td>Please call our customer service representativ...</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



- Ham과 Spam은 약 747개와 4825개로 되어있다.

<br>

### 2.4 Text 길이


```python
messages['length'] = messages['text'].apply(len)
messages.head()
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
      <th>class</th>
      <th>text</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>111</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>155</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>49</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>61</td>
    </tr>
  </tbody>
</table>
</div>



- 단순히 문장의 길이를 length 컬럼으로 생성하여 확인한것

<br>

### 2.5 Spam과 Ham의 문장 길이


```python
messages.hist(column= 'length', by = 'class', bins = 50, figsize=(15, 6))
plt.show()
```


<img src = 'https://user-images.githubusercontent.com/60168331/96459370-04fed480-125d-11eb-97cb-b8776e6dadb8.png'>


- Ham은 길이가 0~200사이에 몰려있고, 긴 것도 있는 반면 Spam은 길이가 100 ~ 170정도의 사이에 몰려있고 긴 것은 없다.

<br>

### 2.6 단어정리를 위한 함수 생성


```python
import string
from nltk.corpus import stopwords

def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    return clean_words
```

- string 패키지에 있는 punctuation은 특수문자를 이야기함
- nopunc는 특수문자를 제거하는 코드
- clean_words는 특수문자를 제거한 nopunc에서 stopwords의 english에 있는 단어가 아닌것만 고른것

<br>

### 2.7 결과 확인


```python
messages['text'].apply(process_text).head()
```




    0    [Go, jurong, point, crazy, Available, bugis, n...
    1                       [Ok, lar, Joking, wif, u, oni]
    2    [Free, entry, 2, wkly, comp, win, FA, Cup, fin...
    3        [U, dun, say, early, hor, U, c, already, say]
    4    [Nah, dont, think, goes, usf, lives, around, t...
    Name: text, dtype: object



- 특수문자는 제거되고, stopword에 없는 단어들만 남게 되었음

<br>

### 2.8 훈련용 데이터 정리


```python
from sklearn.model_selection import train_test_split

msg_train, msg_test, class_train, class_test = train_test_split(messages['text'], messages['class'], test_size = 0.2)
```

- 훈련용 데이터와 테스트용 데이터를 train_test_split 메서드로 사용하여 분리

<br>

### 2.9 Pipe Line 구축


```python
pipeline = Pipeline([
    ('vect', CountVectorizer(analyzer=process_text)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])
```

- 뉴스기사 분석떄 처럼 Pipeline을 구축함

<br>

### 2.10 학습


```python
pipeline.fit(msg_train, class_train)
```




    Pipeline(steps=[('vect',
                     CountVectorizer(analyzer=<function process_text at 0x7f8f263039d0>)),
                    ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])



- pipeline을 통과하여 학습함

<br>

### 2.11 결과


```python
predictions = pipeline.predict(msg_test)
print(classification_report(class_test, predictions))
```

                  precision    recall  f1-score   support
    
             ham       0.96      1.00      0.98       949
            spam       1.00      0.77      0.87       166
    
        accuracy                           0.97      1115
       macro avg       0.98      0.88      0.92      1115
    weighted avg       0.97      0.97      0.96      1115
    


- 와 Acuuracy가 0.97이나 나옴

<br>

### 2.12 Confusion Matrix의 시각화


```python
import seaborn as sns
sns.heatmap(confusion_matrix(class_test, predictions), annot= True, fmt = 'd')
plt.show()
```


<img src = 'https://user-images.githubusercontent.com/60168331/96459372-06300180-125d-11eb-81f4-13b28578cfc5.png'>


- Confusion Matrix를 Heatmap으로 시각화 한것
- 생각보다 잘맞추는 것에 대해 놀랐음

