---
title: MNIST 데이터로 해보는 CNN (Convolution Neral Network)
author: HyunMin Kim
date: 2020-10-29 11:00:00 0000
categories: [Data Science, Deep Learning]
tags: [Tensorflow, Neural Net, CNN, Over Fitting, Convolution Filter, Max Pooling, Drop Out, Pandding]
---

## 1. CNN (Convolution Neral Network)
---
### 1.1 CNN

<img src="https://user-images.githubusercontent.com/60168331/97574180-2fefe200-1a2e-11eb-8044-9427b3672ad7.png">

- 이미지 영상인식의 혁명같은 CNN
- CNN은 이미지의 특징을 검출하여, 분류하는 것

<br>

<img src="https://user-images.githubusercontent.com/60168331/97574332-662d6180-1a2e-11eb-9c63-2e441019be24.png">

- CNN은 특징을 찾는 레이어와 분류를 하는 레이어로 구성됨

<br>

### 1.2 Convolutional Filter

<img src = "https://user-images.githubusercontent.com/60168331/97574899-3e8ac900-1a2f-11eb-8a7c-7e9a9a661611.gif">

- Convolution : 특정 패턴이 있는지 박스로 훑으며 마킹하는 것
- 위 아래선 필터, 좌우선 필터, 대각선 필터, 각종 필터로 해당 패턴이 그림위에 있는지 확인
- 필터는 이미지의 특징을 찾아내기 위한 파라미터 위 그림에서는 주황색의 3 x 3 행렬 (CNN에서 Filter와 Kernel은 같은 의미로 사용됨)
- 필터는 일반적으로 4 x 4 or 3 x 3과 같은 정사각 행렬로 정의됨.
- CNN에서 학습을 통해 필터를 구할 수 있음
- CNN은 입력 데이터를 지정된 간격으로 순회하며 채널별로 합성곱을 하고 모든 채널(컬러의 경우 3개)의 합성곱의 합을 Feature Map로 만듬 
- 위 그림은 채널이 1개인 입력 데이터를 (3, 3) 크기의 필터로 합성곱하는 과정을 나타냄

<br>

### 1.3 Pooling

<img src = "https://user-images.githubusercontent.com/60168331/97575766-8fe78800-1a30-11eb-8239-5bd8f2c3a602.png">
<img src = "https://user-images.githubusercontent.com/60168331/97575782-95dd6900-1a30-11eb-87e9-02001cd26d3c.png">
<img src = "https://user-images.githubusercontent.com/60168331/97575790-9bd34a00-1a30-11eb-8812-fd4106488dce.png">
<img src = "https://user-images.githubusercontent.com/60168331/97575805-a1c92b00-1a30-11eb-9bd7-4a9159ff451b.png">

- 풀링은 점점 더 멀리서 보는것 -> 그림의 크기를 줄이는것

<br>

### 1.4 MaxPooling

<img src = "https://user-images.githubusercontent.com/60168331/97576192-303dac80-1a31-11eb-8269-96affc0ec614.png">

- 그림의 사이즈를 점진적으로 줄이는 법 MaxPooling
- n x n(pool)을 중요한 정보(Max) 한개로 줄임
- 선명한 정보만 남겨서 판단과 학습이 쉬워지고 노이즈가 줄면서 덤으로 융통성도 확보됨
- 4 x 4 행렬  -> 2 x 2 행렬이 됨
- Stride : 좌우로 몇칸씩 이동할지 설정, 보통 2 x 2

<br>

### 1.5 Conv Layer의 의미

<img src = 'https://user-images.githubusercontent.com/60168331/97576328-5fecb480-1a31-11eb-84b7-a3ab0b19e46f.png'>

- Conv : 패턴들을 쌓아가며 점차 복잡한 패턴을 인식
- MaxPooling : 사이즈를 줄여가며, 더욱 추상화 해나감

<br>

### 1.6 CNN 모델 및 코드

<img src = "https://user-images.githubusercontent.com/60168331/97576513-9e826f00-1a31-11eb-89fe-634a6f25a88f.png">

- 위의 내용으로 만든 CNN 모델의 구조와, 파이썬 코드

<br>

### 1.7 Zero Padding

<img src = "https://user-images.githubusercontent.com/60168331/97576784-f1f4bd00-1a31-11eb-9b5c-340d0c394b0e.png">

- Zero Padding : 이미지의 귀퉁이가 짤리니, 사이즈 유지를 위해 Conv 전에 0을 모서리에 추가하고 시작함

<br>

### 1.8 Over Fitting

- 뉴럴넷에 고양이 이미지를 학습 시켰는데, 테스트 이미지가 학습한 이미지와 달라서 제대로 예측하지 못하는 현상
- 즉, 학습 데이터에 과도하게 Fitting되어 있음, 학습 데이터가 아니면 잘 예측하지 못하는것!

<br>

### 1.9 Drop Out

<img src = "https://user-images.githubusercontent.com/60168331/97577225-7a735d80-1a32-11eb-8142-8f41c1820d30.png">
- Overfitting을 방지하기 위한 방법
- 학습 시킬때 일부러 정보를 누락시키거나, 중간 중간에 노드를 끄는것

<br>

## 2. 실습
---
### 2.1 MNIST Data load


```python
from tensorflow.keras import datasets

mnist = datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

X_train = X_train.reshape((60000, 28 ,28, 1))
X_test = X_test.reshape((10000, 28 ,28, 1))
```

- 텐서플로우에서 MNIST 데이터를 불러와서, 데이터 정리
- 255.0으로 나눠준 이유는 이미지가 0 ~ 255 사이의 값을 가지고 있어서, MinMaxScale을 적용한것

<br>

### 2.2 모델 구성


```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                  padding='same', activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_2 (Conv2D)            (None, 28, 28, 32)        832       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 14, 14, 32)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 14, 14, 64)        8256      
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 7, 7, 64)          0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 7, 7, 64)          0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 3136)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 1000)              3137000   
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                10010     
    =================================================================
    Total params: 3,156,098
    Trainable params: 3,156,098
    Non-trainable params: 0
    _________________________________________________________________


<br>

### 2.3 Fit


```python
import time

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

start_time = time.time()

hist = model.fit(X_train, y_train, epochs=5, verbose = 1, validation_data=(X_test, y_test))

print(f'Fit Time :{time.time() - start_time}')
```

    Epoch 1/5
    1875/1875 [==============================] - 35s 19ms/step - loss: 0.1138 - accuracy: 0.9642 - val_loss: 0.0358 - val_accuracy: 0.9877
    Epoch 2/5
    1875/1875 [==============================] - 37s 20ms/step - loss: 0.0467 - accuracy: 0.9853 - val_loss: 0.0315 - val_accuracy: 0.9909
    Epoch 3/5
    1875/1875 [==============================] - 39s 21ms/step - loss: 0.0326 - accuracy: 0.9898 - val_loss: 0.0261 - val_accuracy: 0.9916
    Epoch 4/5
    1875/1875 [==============================] - 40s 21ms/step - loss: 0.0243 - accuracy: 0.9926 - val_loss: 0.0336 - val_accuracy: 0.9893
    Epoch 5/5
    1875/1875 [==============================] - 41s 22ms/step - loss: 0.0223 - accuracy: 0.9930 - val_loss: 0.0264 - val_accuracy: 0.9917
    Fit Time :190.74329090118408


- Accuracy가 0.99...?

<br>

### 2.4 그래프로 보기


```python
import matplotlib.pyplot as plt

plot_target = ['loss' , 'accuracy', 'val_loss', 'val_accuracy']
plt.figure(figsize=(12, 8))

for each in plot_target:
    plt.plot(hist.history[each], label = each)
plt.legend()
plt.grid()
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/97581262-b2c96a80-1a37-11eb-8be7-5a53e5cbb6ec.png'>

- 학습은 아무 문제없이 잘됨.

<br>

### 2.5 Test


```python
score = model.evaluate(X_test, y_test)
print(f'Test Loss : {score[0]}')
print(f'Test Accuracy  : {score[1]}')
```

    313/313 [==============================] - 1s 5ms/step - loss: 0.0264 - accuracy: 0.9917
    Test Loss : 0.02644716575741768
    Test Accuracy  : 0.9916999936103821


- Test도 0.99..
- 틀린 데이터가 궁금해짐

<br>

### 2.6 데이터 예측


```python
import numpy as np

predicted_result = model.predict(X_test)
predicted_labels = np.argmax(predicted_result,  axis=1)
predicted_labels[:10]
```

    array([7, 2, 1, 0, 4, 1, 4, 9, 5, 9])



<br>

### 2.7 틀린 데이터만 모으기


```python
wrong_result = []
for n in range(0, len(y_test)):
    if predicted_labels[n] != y_test[n]:
        wrong_result.append(n)
        
len(wrong_result)
```

    83



- 총 1만개 데이터 중에 83개를 틀림
- 정확도 엄청나다..

<br>

### 2.8 틀린 데이터 16개만 직접 그려보기


```python
import random

samples = random.choices(population=wrong_result, k =16)

plt.figure(figsize=(14, 12))

for idx, n in enumerate(samples):
    plt.subplot(4, 4, idx + 1)
    plt.imshow(X_test[n].reshape(28,28), cmap = 'Greys', interpolation='nearest')
    plt.title('Label ' + str(y_test[n]) + ', Predict ' + str(predicted_labels[n]))
    plt.axis('off')
    
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/97581267-b3fa9780-1a37-11eb-93ff-1f5a614e89af.png'>

- 직접봐도 틀릴만한 것들. 1%

<br>

### 2.9 Model Save


```python
model.save('MNIST_CNN_model.h5')
```

- model.save를 사용하여 만든 모델을 저장할 수 있음!

<br>

## 3. 요약
---
### 3.1 요약

- 이미지, 영상의 최강자 CNN을 튜토리얼 해보았다.
- 사실 하용호님의 자료가 너무 쉽게 잘 정리되어있어서 참 좋았다.
- 딥러닝은 코드를 짜면서도 구성도를 생각해야 해서, 정말 어려운듯 하다.
