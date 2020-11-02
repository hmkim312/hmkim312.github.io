---
title: 특징 추출기 (Feature Extractor)
author: HyunMin Kim
date: 2020-11-02 09:00:00 0000
categories: [Data Science, Deep Learning]
tags: [Tensorflow, Pre Trained Model, Transfer Learning, Inception V3, Feature Extractor]
---

## 1. 특징 추출기
---
### 1.1 특징 추출기
- 미리 훈련된 모델에서는 데이터의 특징만 추출
- 그 특징을 작은 네트워크에 통과시켜 예측하는 방법
- 학습할때 전체 네트워크의 계산을 반복할 필요가 없음

<br>

### 1.2 Inception V3
- Inception은 2014년 구글이 ImageNet이라는 대회에서 GoogleNet이름으로 발표한 CNN 기반의 모델
- <https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4>{:target="_blank"}

<br>

### 1.3 Inception Load


```python
import tensorflow_hub as hub
import tensorflow as tf

inception_url = 'https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4'
feature_model = tf.keras.Sequential([
    hub.KerasLayer(inception_url, output_shape=(2048,), trainable = False)
])
feature_model.build([None, 299, 299, 3])
feature_model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    keras_layer (KerasLayer)     (None, 2048)              21802784  
    =================================================================
    Total params: 21,802,784
    Trainable params: 0
    Non-trainable params: 21,802,784
    _________________________________________________________________


- 파라미터의 갯수가 엄청 많다.

<br>

### 1.4 라벨 불러오기


```python
label_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
label_file = tf.keras.utils.get_file('label', label_url)
label_text = None
with open(label_file, 'r') as f:
    label_text = f.read().split('\n')[:-1]
print(len(label_text))
print(label_text[:10])
print(label_text[-10:])
```

    1001
    ['background', 'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead', 'electric ray', 'stingray', 'cock', 'hen']
    ['buckeye', 'coral fungus', 'agaric', 'gyromitra', 'stinkhorn', 'earthstar', 'hen-of-the-woods', 'bolete', 'ear', 'toilet tissue']


- <https://hmkim312.github.io/posts/사전훈련모델과_전이학습/>{:target="_blank"}
- 사전훈련모델에서 사용한 Label을 다운받음

<br>


```python
import pandas as pd
label_text = pd.read_csv('data/dog_data/labels.csv')
print(label_text.head())
```

                                     id             breed
    0  000bec180eb18c7604dcecc8fe0dba07       boston_bull
    1  001513dfcb2ffafc82cccf4d8bbaba97             dingo
    2  001cdf01b096e06d78e9e5112d419397          pekinese
    3  00214f311d5d2247d5dfe4fe24b2303d          bluetick
    4  0021f9ceb3235effd7fcde7f7538ed62  golden_retriever


- 다운받은 csv는 이런 형태

<br>

### 1.5 ImageDataGenerator 사용 준비


```python
import os
import shutil

os.mkdir('./data/train_sub')

for i in range(len(label_text)):
    if os.path.exists('./data/train_sub/' + label_text.loc[i]['breed']) == False:
        os.mkdir('./data/train_sub/' + label_text.loc[i]['breed'])
    shutil.copy('./data/dog_data/train/' +
                label_text.loc[i]['id'] + '.jpg', './data/train_sub/' + label_text.loc[i]['breed'])
```

- ImageDataGenerator를 사용하기 위한 준비
- ImageDataGenerator는 사전 데이터를 증강하는 기능
- 시간 데이터가 너무 많아서 어쩌면 메모리 부족 현상이 나타날수 있음
- 그럴때 ImageDataGenerator를 이용하면 필요할때마다 디스크에서 배치 크기만큼 조금씩 데이터를 읽을수 있음
- 단, ImageDataGenerator를 사용하려면 각 라벨의 이름을 하위 폴더로 가지고 있도록 해야함

<img src="https://user-images.githubusercontent.com/60168331/97842108-6da08380-1d2a-11eb-9633-c8f62fab5688.png">

<br>

### 1.6 ImageDataGenerator 생성


```python
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

image_size = 299
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1 / 255., horizontal_flip=True, shear_range=0.2,
                                   zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, validation_split=0.25)

valid_datagen = ImageDataGenerator(rescale=1 / 255., validation_split=0.25)
```

- 훈련용 데이터는 픽셀정규화, 좌우반전, 기울이기, 줌, 좌우/상하 평행이동을 함
- 테스트용 데이터는 픽셀 정규화만함

<br>

### 1.7 데이터 생성


```python
train_generator = train_datagen.flow_from_directory(directory='./data/train_sub/', subset='training',
                                                    batch_size=batch_size, seed=13, shuffle=True, class_mode='categorical', target_size=(image_size, image_size))
valid_generator = valid_datagen.flow_from_directory(directory='./data/train_sub/', subset='validation',
                                                    batch_size=1, seed=13, shuffle=True, class_mode='categorical', target_size=(image_size, image_size))
```

    Found 7718 images belonging to 120 classes.
    Found 2504 images belonging to 120 classes.


- 7718장의 train 이미지와 2504장의 test 이미지를 각각 가져왔고, 120개의 class(종)을 가져옴

<br>

### 1.8 데이터의 구조

<img src = 'https://user-images.githubusercontent.com/60168331/97842817-9e34ed00-1d2b-11eb-8637-792ff90cb42b.png'>

<br>

### 1.9 훈련 데이터를 특징 벡터로 변환


```python
batch_step = (7718 * 3) // batch_size
train_features = []
train_Y = []

for idx in range(batch_step):
    if idx % 100 == 0:
        print(idx)
    x, y = train_generator.next()
    train_Y.extend(y)
    
    feature = feature_model.predict(x)
    train_features.extend(feature)
    
train_features = np.array(train_features)
train_Y = np.array(train_Y)
print(train_features.shape)
print(train_Y.shape)
```

    0
    100
    200
    300
    400
    500
    600
    700
    (23084, 2048)
    (23084, 120)


- 시간이 꽤 오래 걸린다.

<br>

### 1.10 검증 데이터에서 특징 벡터 추출


```python
valid_features = []
valid_Y = []

for idx in range(valid_generator.n):
    if idx % 100 == 0:
        print(idx)
        
    x, y = valid_generator.next()
    valid_Y.extend(y)
    
    feature = feature_model.predict(x)
    valid_features.extend(feature)
    
valid_features = np.array(valid_features)
valid_Y = np.array(valid_Y)
print(valid_features.shape)
print(valid_Y.shape)
```

    0
    100
    200
    300
    400
    500
    600
    700
    800
    900
    1000
    1100
    1200
    1300
    1400
    1500
    1600
    1700
    1800
    1900
    2000
    2100
    2200
    2300
    2400
    2500
    (2504, 2048)
    (2504, 120)


- 2504개의 Validation 데이터 특징 추출완료

<br>

### 1.11 모델 생성


```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(2048, )),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(120, activation='softmax')
])

model.compile(tf.optimizers.RMSprop(0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_4 (Dense)              (None, 256)               524544    
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 256)               0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 120)               30840     
    =================================================================
    Total params: 555,384
    Trainable params: 555,384
    Non-trainable params: 0
    _________________________________________________________________


- 파라미터가 처음 2100만개에서 55만개로 줄었음

<br>

### 1.12 Fit


```python
history = model.fit(train_features, train_Y, validation_data=(
    valid_features, valid_Y), epochs=10, batch_size=12)
```

    Epoch 1/10
    1924/1924 [==============================] - 7s 3ms/step - loss: 2.2162 - accuracy: 0.5509 - val_loss: 0.5050 - val_accuracy: 0.8826
    Epoch 2/10
    1924/1924 [==============================] - 7s 4ms/step - loss: 0.6702 - accuracy: 0.8140 - val_loss: 0.3430 - val_accuracy: 0.8938
    Epoch 3/10
    1924/1924 [==============================] - 7s 4ms/step - loss: 0.5090 - accuracy: 0.8479 - val_loss: 0.3194 - val_accuracy: 0.9010
    Epoch 4/10
    1924/1924 [==============================] - 7s 4ms/step - loss: 0.4414 - accuracy: 0.8645 - val_loss: 0.3088 - val_accuracy: 0.9014
    Epoch 5/10
    1924/1924 [==============================] - 7s 4ms/step - loss: 0.4002 - accuracy: 0.8763 - val_loss: 0.3083 - val_accuracy: 0.8998
    Epoch 6/10
    1924/1924 [==============================] - 7s 3ms/step - loss: 0.3608 - accuracy: 0.8872 - val_loss: 0.3105 - val_accuracy: 0.8990
    Epoch 7/10
    1924/1924 [==============================] - 6s 3ms/step - loss: 0.3401 - accuracy: 0.8936 - val_loss: 0.3172 - val_accuracy: 0.9002
    Epoch 8/10
    1924/1924 [==============================] - 6s 3ms/step - loss: 0.3144 - accuracy: 0.9003 - val_loss: 0.3187 - val_accuracy: 0.9010
    Epoch 9/10
    1924/1924 [==============================] - 6s 3ms/step - loss: 0.2909 - accuracy: 0.9076 - val_loss: 0.3164 - val_accuracy: 0.8978
    Epoch 10/10
    1924/1924 [==============================] - 6s 3ms/step - loss: 0.2765 - accuracy: 0.9111 - val_loss: 0.3307 - val_accuracy: 0.8970


- 학습속도가 빠르다.

<br>

### 1.13 학습 상황 그래프


```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8 ))
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/97848879-148a1d00-1d35-11eb-8150-9f8d1bc81f7f.png'>

- 학습이 잘되는건지..

<br>

### 1.14 라벨을 알파벳순으로


```python
unique_Y = label_text['breed'].unique().tolist()
unique_sorted_Y = sorted(unique_Y)
print(unique_sorted_Y)
```

    ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'american_staffordshire_terrier', 'appenzeller', 'australian_terrier', 'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog', 'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick', 'border_collie', 'border_terrier', 'borzoi', 'boston_bull', 'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo', 'doberman', 'english_foxhound', 'english_setter', 'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever', 'french_bulldog', 'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer', 'golden_retriever', 'gordon_setter', 'great_dane', 'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael', 'ibizan_hound', 'irish_setter', 'irish_terrier', 'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier', 'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog', 'mexican_hairless', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'newfoundland', 'norfolk_terrier', 'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog', 'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki', 'samoyed', 'schipperke', 'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'standard_poodle', 'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier', 'whippet', 'wire-haired_fox_terrier', 'yorkshire_terrier']


- 예측을 위해 라벨을 알파벳순으로 정렬함

<br>

### 1.15 예측


```python
import random
import cv2

image_path = random.choice(valid_generator.filepaths)
real_y = image_path.split('/')[3]
idx = unique_sorted_Y.index(real_y)

img = cv2.imread(image_path)
img = cv2.resize(img, dsize = (299, 299))
img = img / 255.0
img = np.expand_dims(img, axis = 0)

feature_vector = feature_model.predict(img)

prediction = model.predict(feature_vector)[0]

top_5_predict = prediction.argsort()[::-1][:5]
labels = [unique_sorted_Y[index] for index in top_5_predict] 
```

- real_y를 구하는 부분에서 해당 경로에 따라 split하여 위치는 다를수 있음

<br>

### 1.16 Top 5예측을 해보자


```python
print(top_5_predict)
print(labels)
```

    [ 10  97  68 105 110]
    ['bedlington_terrier', 'scottish_deerhound', 'lakeland_terrier', 'standard_poodle', 'toy_poodle']


- 저 5개 중에 하나가 맞을지?

<br>

### 1.17 그래프로 보기


```python
plt.figure(figsize = (16,6))

plt.subplot(1,2,1)
plt.imshow(plt.imread(image_path))
plt.title(real_y)
plt.axis('off')

plt.subplot(1,2,2)
color = ['gray'] * 5
if idx in top_5_predict:
    color[top_5_predict.tolist().index(idx)] = 'green'
color = color[::-1]
plt.barh(range(5), prediction[top_5_predict][::-1] * 100, color = color)
plt.yticks(range(5), labels[::-1])
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/97848887-15bb4a00-1d35-11eb-8821-ee5c72c511f8.png'>


- 결과가 매우 잘나온다.

<br>

## 2. 요약
---
### 2.1 요약
- 사전에 훈련된 모델에서 특징만 추출하여, 간단하게 작은 네트워크에 통과하도록 하는 방식이 흥미롭다.
- 결국엔 누군가가 미리 만들어놓은 모델을 이용하여 더 발전 시킬수있고, 또한 이미 훈련이 완료되어있기에 시간도 절약이 된다.
- 딥러닝은 많은 기술과 공부, 그리고 트랜드를 따라가는 노력이 필요한듯 하다
