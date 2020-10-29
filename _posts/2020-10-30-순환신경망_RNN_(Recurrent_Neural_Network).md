---
title: 순환신경망 - RNN (Recurrent Neural Network)
author: HyunMin Kim
date: 2020-10-30 00:00:00 0000
categories: [Data Science, Deep Learning]
tags: [Tensorflow, Recurrent Neural Net, RNN, Simple RNN, LSTM, GRU, Tokenization, Cleaning, Embedding Layer]
---

## 1. Simple RNN
---
### 1.1 순환 신경망

<img src ='https://user-images.githubusercontent.com/60168331/97583622-806d3c80-1a3a-11eb-9135-f7d79096fc65.png'>

- 순서가 있는 데이터를 입력으로 받고 변화하는 입력에 대한 출력을 얻음

<br>

### 1.2 RNN의 한 셀 모양

<img src = 'https://user-images.githubusercontent.com/60168331/97583770-a4c91900-1a3a-11eb-99c5-22db43a40645.png'>

<br>

## 2. RNN 실습
---
### 2.1 간단한 Time Stamp 데이터로 RNN 실습


```python
import tensorflow as tf
import numpy as np

X = []
Y = []

for i in range(6):
    
    lst = list(range(i, i+4))
    
    X.append(list(map(lambda c : [c/10], lst)))
    Y.append((i +4)/10)
    
X = np.array(X)
Y = np.array(Y)

for i in range(len(X)):
    print(X[i], Y[i])
    print()
```

    [[0. ]
     [0.1]
     [0.2]
     [0.3]] 0.4
    
    [[0.1]
     [0.2]
     [0.3]
     [0.4]] 0.5
    
    [[0.2]
     [0.3]
     [0.4]
     [0.5]] 0.6
    
    [[0.3]
     [0.4]
     [0.5]
     [0.6]] 0.7
    
    [[0.4]
     [0.5]
     [0.6]
     [0.7]] 0.8
    
    [[0.5]
     [0.6]
     [0.7]
     [0.8]] 0.9
    


- 0.0 ~ 0.3 -> 0.4
- 0.1 ~ 0.4 -> 0.5
- 이런식의 데이터

<br>

### 2.2 Simple RNN 구성


```python
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(
        units=10, return_sequences=False, input_shape=[4, 1]),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer = 'adam', loss = 'mse')
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    simple_rnn (SimpleRNN)       (None, 10)                120       
    _________________________________________________________________
    dense (Dense)                (None, 1)                 11        
    =================================================================
    Total params: 131
    Trainable params: 131
    Non-trainable params: 0
    _________________________________________________________________


- input_shape이 4.1 이라는것은 timesteps가 4, input_dim이 1
- units : SimpleRNN 레이어에 존재하는 뉴런의 수
- return_sequences : 출력으로 시퀀스 전체를 출력할지 여부

<br>

### 2.3 모델의 구성도

<img src = "https://user-images.githubusercontent.com/60168331/97584902-dee6ea80-1a3b-11eb-8d23-82f123b4df7d.png">

<br>

### 2.4 학습


```python
model.fit(X, Y, epochs=100, verbose=0)
```




    <tensorflow.python.keras.callbacks.History at 0x7fd850d79d90>



<br>

### 2.5 예측


```python
model.predict(np.array([[[-0.1],[0.0],[0.1],[0.2]]]))
```




    array([[0.29263034]], dtype=float32)



- -0.1 ~ 0.2 다음은 0.3이 나와야하는데 0.2926이 나오는것 보니, 비슷은 해보인다

<br>

### 2.6 Simple RNN의 단점

- Long-Term Dependency : 입력 데이터가 길어지면 학습 능력이떨어짐 
- 현재의 답을 얻기 위해 과거의 정보에 의존해야하는 RNN, 하지만 과거 시점이 현재와 너무 멀어지면 문제를 풀기 어려움

<br>

## 3. LSTM
---
### 3.1 LSTM

<img src = "https://user-images.githubusercontent.com/60168331/97585488-86fcb380-1a3c-11eb-8c42-6b6b49bbeb9f.png">

- Simple RNN의 장기 의존성 문제를 해결하기 위한 알고리즘
- Time Step을 가르지르며 셀 상태가 보존

<br>

## 4. LSTM 실습
---
### 4.1 예제


```python
X = []
Y = []

for i in range(1000):
    lst = np.random.rand(100)
    idx = np.random.choice(100, 2, replace = False)
    zeros = np.zeros(100)
    zeros[idx] = 1
    X.append(np.array(list(zip(zeros, lst))))
    Y.append(np.prod(lst[idx]))

print(X[0], Y[0])
```

    [[0.         0.80008966]
     [0.         0.30359938]
     [0.         0.87163122]
     [0.         0.7413119 ]
     [0.         0.11942425]
     [0.         0.27352657]
     [0.         0.76802368]
     [0.         0.14609984]
     [0.         0.18106652]
     [0.         0.01171455]
     [0.         0.92461019]
     [0.         0.27576429]
     [0.         0.8144809 ]
     [0.         0.15831039]
     [0.         0.03956734]
     [0.         0.74086254]
     [0.         0.71514832]
     [0.         0.05864715]
     [0.         0.2165122 ]
     [0.         0.67536621]
     [0.         0.83084592]
     [0.         0.02361978]
     [0.         0.96610312]
     [0.         0.65990591]
     [0.         0.97501121]
     [0.         0.56664119]
     [0.         0.98403786]
     [0.         0.61782982]
     [0.         0.98569084]
     [0.         0.93561593]
     [0.         0.06791456]
     [0.         0.54954407]
     [0.         0.49462747]
     [0.         0.74715515]
     [0.         0.31521783]
     [0.         0.72605368]
     [0.         0.62690249]
     [0.         0.31447398]
     [0.         0.15953186]
     [0.         0.12764518]
     [0.         0.07478073]
     [0.         0.00354316]
     [0.         0.28523369]
     [0.         0.06752979]
     [0.         0.83515363]
     [0.         0.85055375]
     [1.         0.88817727]
     [0.         0.00453563]
     [0.         0.30259626]
     [0.         0.93822272]
     [0.         0.10927959]
     [0.         0.92989588]
     [0.         0.47279259]
     [0.         0.40970746]
     [0.         0.32246528]
     [0.         0.73999216]
     [0.         0.62096274]
     [0.         0.48123822]
     [0.         0.78971826]
     [0.         0.89842873]
     [0.         0.87298911]
     [0.         0.55976035]
     [0.         0.82265406]
     [0.         0.11174719]
     [0.         0.00784555]
     [0.         0.6851193 ]
     [0.         0.75893765]
     [0.         0.50567489]
     [0.         0.01901901]
     [0.         0.7303575 ]
     [0.         0.68753022]
     [0.         0.45555408]
     [0.         0.47891555]
     [0.         0.73691181]
     [0.         0.05961961]
     [0.         0.94850333]
     [0.         0.79596296]
     [0.         0.86432501]
     [0.         0.49509131]
     [0.         0.1899921 ]
     [0.         0.25937904]
     [0.         0.52905918]
     [0.         0.21323525]
     [0.         0.41142003]
     [0.         0.15834983]
     [0.         0.52050195]
     [0.         0.13767634]
     [0.         0.67453866]
     [0.         0.54832447]
     [0.         0.4106969 ]
     [0.         0.57071902]
     [0.         0.54413813]
     [0.         0.16043092]
     [1.         0.37390211]
     [0.         0.6987448 ]
     [0.         0.31205635]
     [0.         0.0487809 ]
     [0.         0.14050364]
     [0.         0.69102483]
     [0.         0.49156883]] 0.3320913581177933


- LSTM을 처음 제안한 논문에서 LSTM의 성능을 확인하기 위해서 제시한 문제

<br>

### 4.2 RNN으로


```python
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(
        units=30, return_sequences=True, input_shape=[100, 2]),
    tf.keras.layers.SimpleRNN(units=30),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    simple_rnn_1 (SimpleRNN)     (None, 100, 30)           990       
    _________________________________________________________________
    simple_rnn_2 (SimpleRNN)     (None, 30)                1830      
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 31        
    =================================================================
    Total params: 2,851
    Trainable params: 2,851
    Non-trainable params: 0
    _________________________________________________________________


- 일단 SimpleRNN으로 확인

<br>

### 4.3 SimpleRNN 훈련 후 그래프


```python
import matplotlib.pyplot as plt

X = np.array(X)
Y = np.array(Y)

history = model.fit(X[:2560], Y[:2560], epochs=100, validation_split=0.2)

plt.plot(history.history['loss'], 'b-', label = 'loss')
plt.plot(history.history['val_loss'], 'r--', label = 'val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
```

    Epoch 1/100
    25/25 [==============================] - 1s 27ms/step - loss: 0.0806 - val_loss: 0.0600
    Epoch 2/100
    25/25 [==============================] - 0s 20ms/step - loss: 0.0517 - val_loss: 0.0586
    Epoch 3/100
    25/25 [==============================] - 0s 19ms/step - loss: 0.0512 - val_loss: 0.0581
    ...
    Epoch 99/100
    25/25 [==============================] - 0s 20ms/step - loss: 0.0100 - val_loss: 0.0821
    Epoch 100/100
    25/25 [==============================] - 0s 19ms/step - loss: 0.0099 - val_loss: 0.0804


<img src = 'https://user-images.githubusercontent.com/60168331/97599977-eadaa880-1a4b-11eb-9d8d-b913de74bbf8.png'>


- Loss가 계속 벌어진다. 이것은 아까 이야기했던 Long-Term Dependency 때문임

<br>

### 4.4 LSTM


```python
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=30, return_sequences=True,
                         input_shape=[100, 2]),
    tf.keras.layers.LSTM(units=30),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer = 'adam', loss = 'mse')
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm (LSTM)                  (None, 100, 30)           3960      
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 30)                7320      
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 31        
    =================================================================
    Total params: 11,311
    Trainable params: 11,311
    Non-trainable params: 0
    _________________________________________________________________

<br>

### 4.5 LSTM 훈련 후 그래프


```python
import matplotlib.pyplot as plt

history = model.fit(X[:2560], Y[:2560], epochs=100, validation_split=0.2)

plt.plot(history.history['loss'], 'b-', label = 'loss')
plt.plot(history.history['val_loss'], 'r--', label = 'val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
```

    Epoch 1/100
    25/25 [==============================] - 1s 46ms/step - loss: 0.0620 - val_loss: 0.0558
    Epoch 2/100
    25/25 [==============================] - 1s 29ms/step - loss: 0.0507 - val_loss: 0.0539
    Epoch 3/100
    25/25 [==============================] - 1s 29ms/step - loss: 0.0508 - val_loss: 0.0543
    ...
    Epoch 100/100
    25/25 [==============================] - 1s 29ms/step - loss: 0.0510 - val_loss: 0.0543


<img src = 'https://user-images.githubusercontent.com/60168331/97599982-eca46c00-1a4b-11eb-9482-c8f7ff9602cb.png'>


- SimpleRNN과는 다르게 Loss와 Val Loss가 같이 떨어진다.

<br>

## 5. GRU 레이어 (Gated Recurrent Unit)
---
### 5.1 GRU의 셀 구조

<img src = 'https://user-images.githubusercontent.com/60168331/97587381-abf22600-1a3e-11eb-9286-cbca34a5b352.png'>

- LSTM에 비해 연산량이 작고, 성능이 어떤 경우는 괜찮은것으로 나타난다.

<br>

### 5.2 GRU 생성


```python
mdoel = tf.keras.Sequential([
    tf.keras.layers.GRU(units=30, return_sequences=True,  input_shape=[100, 2]),
    tf.keras.layers.GRU(units=30),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer = 'adam', loss = 'mse')
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm (LSTM)                  (None, 100, 30)           3960      
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 30)                7320      
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 31        
    =================================================================
    Total params: 11,311
    Trainable params: 11,311
    Non-trainable params: 0
    _________________________________________________________________


<br>

### 5.3 GRU 훈련 후 그래프 그리기


```python
import matplotlib.pyplot as plt

history = model.fit(X[:2560], Y[:2560], epochs=100, validation_split=0.2)

plt.plot(history.history['loss'], 'b-', label = 'loss')
plt.plot(history.history['val_loss'], 'r--', label = 'val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
```

    Epoch 1/100
    25/25 [==============================] - 1s 52ms/step - loss: 0.0506 - val_loss: 0.0550
    Epoch 2/100
    25/25 [==============================] - 1s 29ms/step - loss: 0.0497 - val_loss: 0.0555
    Epoch 3/100
    ...
    Epoch 100/100
    25/25 [==============================] - 1s 30ms/step - loss: 6.9079e-04 - val_loss: 9.3052e-04

<img src = 'https://user-images.githubusercontent.com/60168331/97599985-ed3d0280-1a4b-11eb-8a10-f97e933e2e65.png'>

- 이것도 마찬가지로 Loss와 Val Loss가 엄청 잘 떨어진다.

<br>

## 6. 감성 분석 실습
---
### 6.1 감성분석
- 입력된 자연어 안의 주관적 의견, 감정등을 찾아내는 문제
- 문장의 긍정/부정 등을 구분하는 경우가 많음

<br>

### 6.2 Data Load


```python
import tensorflow as tf

path_to_train_file = tf.keras.utils.get_file('train_txt', 'https://raw.githubusercontent.com/hmkim312/datas/main/navermoviereview/ratings_train.txt')
path_to_test_file = tf.keras.utils.get_file('test_txt', 'https://raw.githubusercontent.com/hmkim312/datas/main/navermoviereview/ratings_test.txt')
```

    Downloading data from https://raw.githubusercontent.com/hmkim312/datas/main/navermoviereview/ratings_train.txt
    14630912/14628807 [==============================] - 4s 0us/step
    Downloading data from https://raw.githubusercontent.com/hmkim312/datas/main/navermoviereview/ratings_test.txt
    4898816/4893335 [==============================] - 2s 0us/step


<br>


```python
train_text = open(path_to_train_file, 'rb').read().decode(encoding='utf-8')
test_text = open(path_to_test_file, 'rb').read().decode(encoding='utf-8')

print(f'Length of text : {len(train_text)} characters')
print(f'Length of text : {len(test_text)} characters')
print()
print(train_text[:100])
```

    Length of text : 6937271 characters
    Length of text : 2318260 characters
    
    id	document	label
    9976970	아 더빙.. 진짜 짜증나네요 목소리	0
    3819312	흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나	1
    10265843


- 일전에 사용했던 영화 리뷰 감성분석 <https://hmkim312.github.io/posts/네이버영화평점을_이용한_감정분석/>{:target="_blank"}
- id: 리뷰한 관객의 id 고유값
- document: 실제 리뷰
- label: 감정 (0: 부정, 1: 긍정)
- 총 200K의 감정분석(20만)
- ratings_test.txt: 5만개의 테스트용 리뷰
- ratings_train.txt: 15만개의 훈련용 리뷰
- 모든 리뷰는 140자 미만
- 100k(10만) 부정 리뷰 (평점이 0점 ~ 4점)
- 100K(10만) 긍정 리뷰 (평점이 9점 ~ 10점)
- 평점이 5점 ~ 8점은 중립리뷰점수로 로 제외시킴

<br>

### 6.3 데이터가 깨끗하지 않음


```python
train_text[:300]
```




    'id\tdocument\tlabel\n9976970\t아 더빙.. 진짜 짜증나네요 목소리\t0\n3819312\t흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나\t1\n10265843\t너무재밓었다그래서보는것을추천한다\t0\n9045019\t교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정\t0\n6483659\t사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 던스트가 너무나도 이뻐보였다\t1\n5403919\t막 걸음마 뗀 3세부터 초등학교 1학년생인 8살용영화.ㅋㅋㅋ...별반개도 아까움.\t0\n7797314\t원작의'



<br>

### 6.4 데이터 전처리


```python
train_text.split('\n')
```

    ['id\tdocument\tlabel',
     '9976970\t아 더빙.. 진짜 짜증나네요 목소리\t0',
     '3819312\t흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나\t1',
     '10265843\t너무재밓었다그래서보는것을추천한다\t0',
     '9045019\t교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정\t0',
     '6483659\t사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 던스트가 너무나도 이뻐보였다\t1',
     ...]


- split을 하여 리뷰별로 가져옴

<br>


```python
train_text.split('\n')[3].split('\t')
```

    ['10265843', '너무재밓었다그래서보는것을추천한다', '0']


- 리뷰 한줄을 탭으로 split하면 id, review, target이 나옴

<br>

### 6.5 Target 데이터 정리


```python
import numpy as np

train_Y = np.array([[int(row.split('\t')[2])] for row in train_text.split('\n')[1:] if row.count('\t') >0])
test_Y = np.array([[int(row.split('\t')[2])] for row in test_text.split('\n')[1:] if row.count('\t') >0])

print(train_Y.shape, test_Y.shape)
print(train_Y[:5])
```

    (150000, 1) (50000, 1)
    [[0]
     [1]
     [0]
     [0]
     [1]]


- 일단 타겟이 되는 0(부정), 1(긍정) 데이터를 따로 모아놨음

<br>

### 6.6 Tokenization, Cleaning

- Tokenization : 자연어 처리 가능한 최소의 단위로 나누는 것, 이번에는 띄어쓰기
- Cleaning : 불필요한 기호를 제거

<br>

### 6.7 Cleaning 함수 생성 및 데이터 전처리


```python
import re
def clean_str(string):
    string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\'{2,}", "\'", string)
    string = re.sub(r"\'", "", string)
    
    return string.lower()
```

- 불필요한 특수문자를 제거하는 함수 생성

<br>


```python
train_text_X = [row.split('\t')[1] for row in train_text.split('\n')[1:] if row.count('\t') > 0]
train_text_X = [clean_str(sentence) for sentence in train_text_X]
sentences = [sentence.split(' ') for sentence in train_text_X]
for i in range(5):
    print(sentences[i])
```

    ['아', '더빙', '진짜', '짜증나네요', '목소리']
    ['흠', '포스터보고', '초딩영화줄', '오버연기조차', '가볍지', '않구나']
    ['너무재밓었다그래서보는것을추천한다']
    ['교도소', '이야기구먼', '솔직히', '재미는', '없다', '평점', '조정']
    ['사이몬페그의', '익살스런', '연기가', '돋보였던', '영화', '!', '스파이더맨에서', '늙어보이기만', '했던', '커스틴', '던스트가', '너무나도', '이뻐보였다']


- train_text 리뷰에 '\t'가 있다면 split하는 코드
- 위에서 만든 clean_str을 적용하는 코드
- 공백(띄어쓰기)로 split하는 코드

<br>

### 6.8 리뷰의 길이


```python
import matplotlib.pyplot as plt

sentence_len = [len(sentence) for sentence in sentences]
sentence_len.sort()
plt.plot(sentence_len)
plt.show()

print('리뷰의 길이가 25자 미만인것 : ', sum([int(l <= 25)for l in sentence_len]))
```

<img src = 'https://user-images.githubusercontent.com/60168331/97599989-edd59900-1a4b-11eb-92bc-eae216a8f9bc.png'>


    리뷰의 길이가 25자 미만인것 :  142587


- 학습을 위해 네트워크에 입력을 넣을땐 입력 데이터는 그 크기가 같아야함
- 입력 벡터의 크기를 맞추기위해 긴 문장을 줄이고, 짧은 문장은 공백으로 채우는 방법을 사용
- 15만개의 문장중에 대부분이 25단어 이하로 되어있음, 142587개

<br>

### 6.9 데이터 전처리 25개 단어까지 넣기


```python
sentences_new = []
for sentence in sentences:
    sentences_new.append([word[:5] for word in sentence][:25])
    
sentences = sentences_new
for i in range(5):
    print(sentences_new[i])
```

    ['아', '더빙', '진짜', '짜증나네요', '목소리']
    ['흠', '포스터보고', '초딩영화줄', '오버연기조', '가볍지', '않구나']
    ['너무재밓었']
    ['교도소', '이야기구먼', '솔직히', '재미는', '없다', '평점', '조정']
    ['사이몬페그', '익살스런', '연기가', '돋보였던', '영화', '!', '스파이더맨', '늙어보이기', '했던', '커스틴', '던스트가', '너무나도', '이뻐보였다']


<br>

### 6.10 토크나이징과 패딩


```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(sentences)
train_X = tokenizer.texts_to_sequences(sentences)
train_X = pad_sequences(train_X, padding= 'post')
print(train_X[:3])
```

    [[  25  884    8 5795 1111    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0]
     [ 588 5796 6697    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0]
     [   0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0]]


<br>

### 6.11 모델 구성


```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(20000, 300, input_length = 25),
    tf.keras.layers.LSTM(units = 50),
    tf.keras.layers.Dense(2, activation = 'softmax')
])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 25, 300)           6000000   
    _________________________________________________________________
    lstm_2 (LSTM)                (None, 50)                70200     
    _________________________________________________________________
    dense_4 (Dense)              (None, 2)                 102       
    =================================================================
    Total params: 6,070,302
    Trainable params: 6,070,302
    Non-trainable params: 0
    _________________________________________________________________


<br>

### 6.12 임베딩 레이어(Embedding Layer)

- 임베딩 레이어 : 자연어를 수치화된 정보로 바꾸기 위한 레이어
- 자연어는 시간의 흐름에 따라 정보가 연속적으로 이어지는 스퀀스 데이터
- 영어는 문자 단위, 한글은 문자를 넘어 자소 단위로도 쪼개기도 함, 혹은 형태소, 띄어쓰기로도 하기도함
- 여러 단어로 묶어서 사용하는 n-gram 방식도 있음
- 원핫인코딩까지 포함

<br>

### 6.13 학습


```python
history = model.fit(train_X, train_Y, epochs=5, batch_size = 120, validation_split= 0.2)
```

    Epoch 1/5
    1000/1000 [==============================] - 50s 50ms/step - loss: 0.4301 - accuracy: 0.7884 - val_loss: 0.3805 - val_accuracy: 0.8224
    Epoch 2/5
    1000/1000 [==============================] - 51s 51ms/step - loss: 0.3263 - accuracy: 0.8467 - val_loss: 0.3903 - val_accuracy: 0.8211
    Epoch 3/5
    1000/1000 [==============================] - 47s 47ms/step - loss: 0.2766 - accuracy: 0.8662 - val_loss: 0.4175 - val_accuracy: 0.8196
    Epoch 4/5
    1000/1000 [==============================] - 47s 47ms/step - loss: 0.2357 - accuracy: 0.8836 - val_loss: 0.4570 - val_accuracy: 0.8113
    Epoch 5/5
    1000/1000 [==============================] - 47s 47ms/step - loss: 0.2016 - accuracy: 0.8990 - val_loss: 0.5628 - val_accuracy: 0.8102


<br>

### 6.14 테스트


```python
test_sentence = '재미있을 줄 알았는데 완전 실망했다. 너무 졸리고 돈이 아까웠다'
test_sentence = test_sentence.split(' ')
test_sentences =[]
now_sentence = []
for word in test_sentence:
    now_sentence.append(word)
    test_sentences.append(now_sentence[:])
    
test_X_1 = tokenizer.texts_to_sequences(test_sentences)
test_X_1 = pad_sequences(test_X_1, padding = 'post', maxlen=25)
prediction = model.predict(test_X_1)
for idx, sentence in enumerate(test_sentences):
    print(sentence)
    print(prediction[idx])
```

    ['재미있을']
    [0.33282876 0.66717124]
    ['재미있을', '줄']
    [0.3296478  0.67035216]
    ['재미있을', '줄', '알았는데']
    [0.41936734 0.5806327 ]
    ['재미있을', '줄', '알았는데', '완전']
    [0.36908486 0.6309151 ]
    ['재미있을', '줄', '알았는데', '완전', '실망했다.']
    [0.36908486 0.6309151 ]
    ['재미있을', '줄', '알았는데', '완전', '실망했다.', '너무']
    [0.375071 0.624929]
    ['재미있을', '줄', '알았는데', '완전', '실망했다.', '너무', '졸리고']
    [0.989423   0.01057698]
    ['재미있을', '줄', '알았는데', '완전', '실망했다.', '너무', '졸리고', '돈이']
    [0.9979898  0.00201018]
    ['재미있을', '줄', '알았는데', '완전', '실망했다.', '너무', '졸리고', '돈이', '아까웠다']
    [0.9983991  0.00160098]


- 처음에는 `재미있을` 이라는 단어 떄문에 긍정으로 판단되다가, `졸리고`가 나오자 바로 부정으로 바뀌었다.
- 앞에가 0(부정), 뒤에가 1(긍정)일 확률임

<br>

## 7. 요약
---
### 7.1 요약
- RNN의 종류와 이를 활용하여 감성분석을 진행했다.
- 솔직히 아직 공부가 부족해서 무슨 이야기인지 이해가 안가는 부분이 있음
- 딥러닝은 따로 인강을 들으며 더 공부를 해야겠다.
