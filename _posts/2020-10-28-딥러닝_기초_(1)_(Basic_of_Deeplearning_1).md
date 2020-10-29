---
title: 딥러닝 기초 (1) (Basic of Deeplearning 1)
author: HyunMin Kim
date: 2020-10-28 10:00:00 0000
categories: [Data Science, Deep Learning]
tags: [Tensorflow, Neural Net, Linear Regression, Loss Function, Activation Function, GD, SGD, Adam, Softmax, ]
---

## 1. Tensorflow 설치
---
### 1.1 Pip Upgrade
- pip install --upgrade pip
- Tensorflow 설치를 위해선 pip 버전이 19.X 이상이어야함

<br>

### 1.2 Tensorflow 설치
- pip install tensorflow

<br>

## 2. Tensorflow
---
### 2.1 Tensorflow란?

- 머신러닝을 위한 오픈소스 플랫폼 - 딥러닝 프레임 워크
- 구글이 주도적으로 개발하였고, 구글코랩에는 기본으로 설치되어 있음
- 최근 2.x 버전이 발표되었음
- Keras와 병합되었음
- Tensor : 벡터나 행렬을 의미
- Graph : 텐서가 흐르는 경로 or 공간
- Tensor Flow : Tensor가 Graph를 통해 흐름

<br>

## 3. 딥러닝 기초
---
### 3.1 Neural Net

<img src="https://user-images.githubusercontent.com/60168331/97377474-7ac10b00-1903-11eb-8f71-1f4d729f6f81.png">

- Neural Net은 신경망에서 아이디어를 얻어서 시작됨

<br>

### 3.2 뉴런

<img src="https://user-images.githubusercontent.com/60168331/97377590-beb41000-1903-11eb-884b-c7346a45efd0.png">

- 뉴런은 입력, 가중치, 활성화함수, 출력으로 구성되어 있음
- 뉴런에서 학습할때 변하는 것은 가중치, 처음에는 초기화를 통해 랜덤값을 넣고, 학습과정에서 일정한 값으로 수렴됨

<br>

### 3.3 레이어와 망

<img src="https://user-images.githubusercontent.com/60168331/97377681-ef944500-1903-11eb-9c09-cd85c339656c.png">

- 뉴런이 모여서 Layer를 구성하고, 망(net)이 됨

<br>

### 3.4 딥러닝

<img src="https://user-images.githubusercontent.com/60168331/97377780-28341e80-1904-11eb-9002-45b14644a9c1.png">

- 이러한 신경망이 많아지면 Deep Learning이 됨

<br>

## 4. 실습 1
---
### 4.1 Blood Fat


```python
import tensorflow as tf
tf.__version__
```




    '2.3.1'



- Tensorflow 버전 확인

<br>


```python
import numpy as np

raw_data = np.genfromtxt('https://raw.githubusercontent.com/hmkim312/datas/main/blood%20fat/x09.txt', skip_header=36)
raw_data
```




    array([[  1.,   1.,  84.,  46., 354.],
           [  2.,   1.,  73.,  20., 190.],
           [  3.,   1.,  65.,  52., 405.],
           [  4.,   1.,  70.,  30., 263.],
           [  5.,   1.,  76.,  57., 451.],
           [  6.,   1.,  69.,  25., 302.],
           [  7.,   1.,  63.,  28., 288.],
           [  8.,   1.,  72.,  36., 385.],
           [  9.,   1.,  79.,  57., 402.],
           [ 10.,   1.,  75.,  44., 365.],
           [ 11.,   1.,  27.,  24., 209.],
           [ 12.,   1.,  89.,  31., 290.],
           [ 13.,   1.,  65.,  52., 346.],
           [ 14.,   1.,  57.,  23., 254.],
           [ 15.,   1.,  59.,  60., 395.],
           [ 16.,   1.,  69.,  48., 434.],
           [ 17.,   1.,  60.,  34., 220.],
           [ 18.,   1.,  79.,  51., 374.],
           [ 19.,   1.,  75.,  50., 308.],
           [ 20.,   1.,  82.,  34., 220.],
           [ 21.,   1.,  59.,  46., 311.],
           [ 22.,   1.,  67.,  23., 181.],
           [ 23.,   1.,  85.,  37., 274.],
           [ 24.,   1.,  55.,  40., 303.],
           [ 25.,   1.,  63.,  30., 244.]])



- 해당 데이터는 고혈압(혈중 지질)을 나타낸 데이터
- 3번째는 몸무게, 4번째는 나이, 마지막이 타겟인 혈중지질임

<br>

### 4.2 그래프로 보기


```python
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
%matplotlib inline

xs = np.array(raw_data[:,2], dtype = np.float32)
ys = np.array(raw_data[:,3], dtype = np.float32)
zs = np.array(raw_data[:,4], dtype = np.float32)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(xs, ys, zs)
ax.set_xlabel('Weight')
ax.set_ylabel('Age')
ax.set_zlabel('Blood Fat')
ax.view_init(15, 15)
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/97444837-ee98fd00-196f-11eb-85db-91e5ab4d8826.png'>

- 나이와 몸무게를 알려주면 주어진 데이터 기준의 Blood Fat을 얻는것
- 즉, 46살 84키로인 사람의 데이터 기준 Blood Fat을 물으면 답이 나와야하는것
- Linear Regression으로 풀자

<br>

### 4.3 Linear Regression

<img src="https://user-images.githubusercontent.com/60168331/97378701-5286db80-1906-11eb-9e2c-f273bffee571.png">

- 직선 모델을 얻는것으로 하면, 주어진 입출력 데이터로 W와 b 즉, 모델을 얻어야 함

<br>

### 4.4 예측

<img src="https://user-images.githubusercontent.com/60168331/97378809-9aa5fe00-1906-11eb-8ad4-1fc304c9d6c1.png">

- 모델 (W,b)를 이용해서, 예측을 한다. 즉 age 40, weight 80인 사람의 y(blood fat)은 얼마인가?

<br>

### 4.5 목표

<img src="https://user-images.githubusercontent.com/60168331/97378879-cde88d00-1906-11eb-8edf-6fd2cda8ab4d.png">

- 목적은 x1, x2를 입력해서 y가 나오게 하는 W와 b를 구하는것

<br>

### 4.6 데이터 정리


```python
x_data = np.array(raw_data[:, 2:4], dtype=np.float32)
y_data = np.array(raw_data[:, 4], dtype=np.float32)
y_data = y_data.reshape((25, 1))
```

- 원래 데이터에서 x와 y를 정리하고, y를 reshape해준다

<br>

### 4.7 모델 생성


```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=(2,)),
])

model.compile(optimizer='rmsprop', loss='mse')
```

<br>

#### 4.7.1 Loss

- 학습을 위해서는 loss(cost) 함수를 정해야함
- 정답까지 얼마나 멀리 있는지 측정하는 함수
- mse : 오차 제곱의 평균
- 그외 선택 가능한 loss <https://keras.io/api/losses/>{:target="_blank"}

<br>

### 4.7.2 Optimizer

- Optimizer를 선정함, loss를 어떻게 줄일것인지를 결정하는 것
- loss함수를 최소화하는 가중치를 찾아가는 과정에 대한 알고리즘
- 그외 선택 가능한 Optimizer <https://keras.io/api/optimizers/>{:target="_blank"}

<br>

### 4.8 Summary


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 1)                 3         
    =================================================================
    Total params: 3
    Trainable params: 3
    Non-trainable params: 0
    _________________________________________________________________


- 모델 구성에 대한 요약을 볼수 있다.
- 현재는 모델 구성만 한것이고, 학습을 시도한것은 아니다.
- 나이와 몸무게를 받아서 Blood Fat을 추정하는 모델을 학습을 통해 얻으려고 함
- 모델(네트워크)를 구성하였고, 모델의 loss function을 선정, loss의 감소를 위한 optimizer도 선정함

<br>

### 4.9 모델 학습


```python
hist = model.fit(x_data, y_data, epochs=5000)
```

    Epoch 1/5000
    1/1 [==============================] - 0s 647us/step - loss: 112005.7578
    Epoch 2/5000
    1/1 [==============================] - 0s 469us/step - loss: 111776.2500
    Epoch 3/5000
    1/1 [==============================] - 0s 544us/step - loss: 111609.9688
    ...
    Epoch 4999/5000
    1/1 [==============================] - 0s 386us/step - loss: 2318.3562
    Epoch 5000/5000
    1/1 [==============================] - 0s 486us/step - loss: 2317.8252

- epochs = 1은 전체 데이터 셋에 대해 한 번 학습을 완료한 상태
- epochs = 40이라면 전체 데이터를 40번 사용해서 학습을 거치는 것

<br>

### 4.10 Loss 확인


```python
plt.plot(hist.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()
```

<img src ='https://user-images.githubusercontent.com/60168331/97444847-f0fb5700-196f-11eb-9512-29f2bcb8dc0d.png'>


- Epochs를 진행할수록 Loss가 떨어짐을 알수있다.
- 처음에는 140,000 였다가 마지막에 2000대로 떨어졌다.

<br>

### 4.11 예측


```python
model.predict(np.array([100,44]).reshape(1,2))
```




    array([[395.40054]], dtype=float32)



- 몸무게 100에 44살인 사람의 Blood Fat을 예측하니 387.94가 나왔음

<br>


```python
model.predict(np.array([60,25]).reshape(1,2))
```




    array([[233.83997]], dtype=float32)



- 몸무게 60에 25살 사람의 Blood Fat은 228.48이 나옴

<br>

### 4.12 W(가중치)와 bias는?


```python
W_, b_ = model.get_weights()
print(f'가중치(Weight) is : {W_})')
print(f'Bias is : {b_})')
```

    가중치(Weight) is : [[2.211234]
     [3.847958]])
    Bias is : [4.9669867])


- 가중치와 bias는 get_weights를 통해서 알수 있음

<br>

### 4.13 모델 확인 및 그리기


```python
x1 = np.linspace(20, 100, 50).reshape(50, 1)
x2 = np.linspace(10, 70, 50).reshape(50, 1)

X = np.concatenate((x1, x2), axis=1)
y = np.matmul(X, W_) + b_
```

- x1은 몸무게, x2는 나이, y는 모델의 값을 적용하여 만들어낸 Blood Fat, 총 50개의 데이터

<br>


```python
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(xs, ys, zs) # 기존 데이터
ax.scatter(x1, x2, y) # 새로 만든 데이터
ax.set_xlabel('Weight')
ax.set_ylabel('Age')
ax.set_zlabel('Blood Fat')
ax.view_init(15,15)
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/97444850-f193ed80-196f-11eb-8dbc-ec67c64a9eaa.png'>


- 주황색이 새로 만든 데이터로 그린 스캐터플롯임
- 일직선이 나오는걸로 봐서 생각보다 모델이 잘 만들어진듯 하다

<br>

## 5. XOR 문제
---
### 5.1 XOR 문제

<img src="https://user-images.githubusercontent.com/60168331/97380913-579a5980-190b-11eb-88ba-c4ef29300493.png">

- 입력이 같으면 `0`, 다르면 `1`의 출력이 나오는 소자
- 즉, 입력 중 어느 하나 만 `1`일 경우에 만 출력이 `1`이 되는 소자

<br>

### 5.2 선형 모델의 XOR

<img src="https://user-images.githubusercontent.com/60168331/97381045-9a5c3180-190b-11eb-8ba3-50186c51ae95.png">

- 선형 모델로는 XOR 문제를 풀수 없음

<br>

### 5.3 데이터 준비


```python
import numpy as np

X = np.array([[0, 0],
              [1, 0],
              [0, 1],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])
```

- XOR의 진리표에 따라 A, B가 같지 않으면 1, 같으면 0을 출력

<br>

### 5.4 모델 생성


```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation = 'sigmoid', input_shape = (2,)),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])
```

<img src="https://user-images.githubusercontent.com/60168331/97381336-38e89280-190c-11eb-8a09-5d656c2b56a4.png">

- 위 모델의 생김새는 그림과 같다

<br>

### 5.5 Model의 Compile


```python
model.compile(optimizer=tf.keras.optimizers.SGD(lr = 0.1), loss = 'mse')
```

- 옵티마이저를 선정하고 학습률을 선정함
- Loss는 mse로
- SGD : 그래디언트 벡터
- lr : 아래로 내려가는 정도

<br>

### 5.6 Model Summary


```python
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 2)                 6         
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 3         
    =================================================================
    Total params: 9
    Trainable params: 9
    Non-trainable params: 0
    _________________________________________________________________


<br>

### 5.7 학습


```python
hist = model.fit(X, y, epochs=5000, batch_size = 1)
```

    Epoch 1/5000
    4/4 [==============================] - 0s 611us/step - loss: 0.2555
    Epoch 2/5000
    4/4 [==============================] - 0s 568us/step - loss: 0.2555
    Epoch 3/5000
    4/4 [==============================] - 0s 535us/step - loss: 0.2555
    ...
    Epoch 4918/5000
    4/4 [==============================] - 0s 601us/step - loss: 0.0032
    Epoch 4919/5000
    4/4 [==============================] - 0s 528us/step - loss: 0.0032
    Epoch 4920/5000
    4/4 [==============================] - 0s 526us/step - loss: 0.0032


- Epochs : 지정된 횟수만큼 학습 하는것
- Batch_size : 한번의 학습에 사용될 데이터의 수
- fit을 여러번 진행하면 처음부터 진행하는것이 아닌, 기존에 업데이트된 loss에서부터 계속 연속으로 지정됨

<br>

### 5.8 학습결과


```python
model.predict(X)
```




    array([[0.05735993],
           [0.94680035],
           [0.9466035 ],
           [0.05780149]], dtype=float32)



- 0은 아니지만 0과 근접한 수치, 1은 아니지만 1과 근접한 수치가 나옴

<br>

### 5.9 Loss Graph


```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 12))
plt.plot(hist.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/97444853-f22c8400-196f-11eb-9661-26ddef35f6bd.png'>


- Epochs가 진행되면서 loss가 쭉 떨어지다가 다시 변화가 없음

<br>

### 5.10 가중치 확인


```python
for w in model.weights:
    print('----')
    print(w)
    print()
```

    ----
    <tf.Variable 'dense_1/kernel:0' shape=(2, 2) dtype=float32, numpy=
    array([[5.907916 , 3.734393 ],
           [5.844629 , 3.7225044]], dtype=float32)>
    
    ----
    <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([-2.4336848, -5.7000146], dtype=float32)>
    
    ----
    <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
    array([[ 7.470416],
           [-8.074545]], dtype=float32)>
    
    ----
    <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([-3.3748236], dtype=float32)>
    


<br>

## 6. 분류 실습
---
### 6.1 Iris 데이터


```python
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target
```

- Iris 데이터를 불러오고 X,y로 나누어줌
- 다만 y가 0,1,2로 나누어져 있어서 원핫인코딩이 필요하다

<br>

### 6.2 원핫인코딩 

<img src="https://user-images.githubusercontent.com/60168331/97438590-564b4a00-1968-11eb-9a08-6d31f0c15d6d.png">
- 타켓을 1,2,3을 각 컬럼으로 만들어 0과 1로 만드는 행위

<br>

### 6.3 Sklearn의 One Hot Encoder


```python
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
enc.fit(y.reshape(len(y), 1))
```




    OneHotEncoder(handle_unknown='ignore', sparse=False)



- Sklearn의 One Hot Encoder를 사용하여 재정리를 해줌



```python
enc.categories_
```




    [array([0, 1, 2])]



- Target의 컬럼

<br>


```python
y_onehot = enc.transform(y.reshape(len(y), 1))
y_onehot[:3]
```




    array([[1., 0., 0.],
           [1., 0., 0.],
           [1., 0., 0.]])



- fit한것을 Transform 함

<br>

### 6.4 데이터 정리


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size = 0.2, random_state = 13)
```

- 데이터를 학습과 테스트데이터로 나눔

<br>

### 6.5 신경망 구조도

<img src="https://user-images.githubusercontent.com/60168331/97439224-3e27fa80-1969-11eb-9982-fa8925540bca.png">

- Iris 데이터의 구조도

<br>

### 6.6 코드로 작성


```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, input_shape=(4,), activation = 'relu'),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dense(3, activation = 'softmax'),
])
```

<br>

### 6.7 Activation이란

<img src="https://user-images.githubusercontent.com/60168331/97440349-b17e3c00-196a-11eb-8e7f-279ae965ebb0.png">

- 하나의 뉴런 끝단에 Activation이라는 함수가 있음

<br>

### 6.9 역전파(Back-Propagation)

<img src="https://user-images.githubusercontent.com/60168331/97440610-03bf5d00-196b-11eb-8a72-8ef08b9fb617.png">

<br>

### 6.10 역전파에는 sigmoid문제가 있음

<img src="https://user-images.githubusercontent.com/60168331/97440698-2487b280-196b-11eb-8bb8-64fc092c34ea.png">

<br>

### 6.11 Vanishing Gradient 현상

<img src="https://user-images.githubusercontent.com/60168331/97440796-497c2580-196b-11eb-9ae9-195ca76c5941.png">

<br>

### 6.12 Relu

<img src="https://user-images.githubusercontent.com/60168331/97440914-729cb600-196b-11eb-9397-42c9a088a7f4.png">

<br>

### 6.13 Softmax?

<img src="https://user-images.githubusercontent.com/60168331/97441040-99f38300-196b-11eb-80e7-6ea498c9614d.png">

<br>

### 6.14 완성된 모델

<img src="https://user-images.githubusercontent.com/60168331/97441146-bc859c00-196b-11eb-8bd4-892048344a97.png">

<br>

### 6.15 Model Summary


```python
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_3 (Dense)              (None, 32)                160       
    _________________________________________________________________
    dense_4 (Dense)              (None, 32)                1056      
    _________________________________________________________________
    dense_5 (Dense)              (None, 32)                1056      
    _________________________________________________________________
    dense_6 (Dense)              (None, 3)                 99        
    =================================================================
    Total params: 2,371
    Trainable params: 2,371
    Non-trainable params: 0
    _________________________________________________________________

<br>

### 6.16 Gradient Decent

- Grdient Decent : 기준 뉴럴넷이 가중치 Parameter들을 최적화(Optimize)하는 방법
- Loss Function의 현 가중치에서 기울기(Gradient)를 구해서 Loss를 줄이는 방향으로 업데이트

<br>

### 6.17 Loss Function

- 뉴럴넷은 Loss(or Cost) Function을 가지고 있음. = 틀린 정도
- 현재 가진 Weight 세팅에서 내가 가진 데이터를 다 넣으면 전체 에러가 계산됨
- 거기서 미분을 하면 에러를 줄이는 방향을 알수 있음 (내자리의 기울기 * 반대 방향)
- 그 방향으로 정해진 스텝량(Learning Rate)을 곱해서 weight를 이동
- 위의 내용을 반복
- Weight의 업데이트 = 에러 낮추는 방향(Decent) * 한발자국 크기(Learning Rate) * 현 지점의 기울기 (Gradient)

<br>

### 6.18 SGD (Stochastic Gradient Decent)

- Gradient Decent : Full Batch로 학습데이터를 모두 다 읽고 최적의 1스텝을 감
- SGD (Stochastic Gradient Decent) : 학습 데이터를 토막(Mini Batch)내서 토막낸 학습데이터로 1스탭씩 가는것

<br>

### 6.19 GD vs SGD

- Gradient Decent : 모든걸 계산 후 최적의 1 스탭을 감. 최적인데 너무 느림!
- Stochastic Gradient Descent : 일부만 검토 후 1 스탭을 감. 최적은 아니지만 매우 빠름 

<br>

### 6.20 Optimizer의 선택

- 산을 잘타고 내려오는것은 어느 방향으로 발을 디딜지, 얼마의 보폭으로 발을 디딜지 두 가지를 잘 잡아야 빠르게 타고 내려옴
- SGD를 개선한 Opmimizer들고 있음

<br>
    
### 6.21 Optimizer의 발달 계보

<img src="https://user-images.githubusercontent.com/60168331/97443527-7a118e80-196e-11eb-9831-f8ed418def5a.png">

- 일단 데이터가 복잡하다면 Adam을 쓴다.

<br>

### 6.22 학습


```python
hist = model.fit(X_train, y_train, epochs=100)
```

    Epoch 1/100
    4/4 [==============================] - 0s 777us/step - loss: 0.9711 - accuracy: 0.4583
    Epoch 2/100
    4/4 [==============================] - 0s 635us/step - loss: 0.8974 - accuracy: 0.7417
    Epoch 3/100
    4/4 [==============================] - 0s 584us/step - loss: 0.8529 - accuracy: 0.6583
    ...
    Epoch 100/100
    4/4 [==============================] - 0s 589us/step - loss: 0.0624 - accuracy: 0.9833


- 100번의 Epochs를 했고 Accuracy가 높게 잘 나온다.

<br>

### 6.23 Test 데이터의 Accuracy


```python
model.evaluate(X_test, y_test, verbose=2)
```

    1/1 - 0s - loss: 0.0932 - accuracy: 0.9667
    [0.09320703893899918, 0.9666666388511658]



<br>

### 6.24 Loss와 Acc의 변화


```python
plt.figure(figsize=(12,6))
plt.plot(hist.history['loss'])
plt.plot(hist.history['accuracy'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/97444857-f22c8400-196f-11eb-8418-4211b16892b9.png'>

- Epochs가 늘어나며 Loss는 떨어지고, Accuracy는 일정 구간까지 올라가는 모습을 보인다.

<br>

## 7. 요약
---
### 7.1 요약

- 딥러닝의 기초를 살짝 다루었다.
- Optimizer와 Loss Function, Activation Function 등 생소한 표헌과 잘 이해가지 않는 함수들이 대거 출연하여 어려운 내용이었다.
- 하용호님의 자료 <https://www.slideshare.net/yongho/ss-79607172>{:target="_blank"}를 참조 하였다.
- 좀더 공부를 해야겠다.