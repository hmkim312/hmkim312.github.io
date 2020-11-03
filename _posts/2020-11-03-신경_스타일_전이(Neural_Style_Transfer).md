---
title: 신경 스타일 전이 (Neural Style Transfer)
author: HyunMin Kim
date: 2020-11-03 09:00:00 0000
categories: [Data Science, Deep Learning]
tags: [Tensorflow, Pre Trained Model, Transfer Learning, VGG19, Neual Style Transfer, Gram Matrix]
---

## 1. 신경 스타일 전이
---
### 1.1 신경 스타일 전이

<img src = 'https://user-images.githubusercontent.com/60168331/97957978-9dfe2580-1def-11eb-9be3-6c9e6aa1bc63.png'>

- 2015년 딥러닝과 예술의 만남으로 큰 화제가 됨
- 그림의 스타일을 학습하여, 사진에 전이시킴
- 고흐 <별이 빛나는 밤에> 작품의 스타일을 학습해서, 사진에 전이를 시킨것

<br>

### 1.2 텍스쳐 합성

<img src = 'https://user-images.githubusercontent.com/60168331/97958180-0cdb7e80-1df0-11eb-8592-8482a02f4901.png'>

- 한 장의 이미지를 원본으로 삼아 해당 텍스쳐를 재생성하는 작업

<br>

## 2. 텍스쳐 실습
---
### 2.1 텍스쳐 이미지 불러오기


```python
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

style_path = tf.keras.utils.get_file('style.jpg', 'http://bit.ly/2mGfZIq')

style_image = plt.imread(style_path)
style_image = cv2.resize(style_image, dsize =(224, 224))
style_image = style_image / 255.0
plt.imshow(style_image)
plt.show()
```

<img src ='https://user-images.githubusercontent.com/60168331/97981995-779fb080-1e16-11eb-85ff-a8e00862f59e.png'>


- 이렇게 생긴 이미지의 스타일을 텍스쳐 합성 해볼것

<br>

### 2.2 Target Image 생성


```python
target_image = tf.random.uniform(style_image.shape)
print(target_image[0,0,:])
plt.imshow(target_image)
plt.show()
```

    tf.Tensor([0.09172022 0.59934413 0.2797972 ], shape=(3,), dtype=float32)


<img src = 'https://user-images.githubusercontent.com/60168331/97982004-7a9aa100-1e16-11eb-89bd-c014209ef41e.png'>


- 노이즈가 섞인(옛날 티비 끝나고 나오는 듯한 사진) 이미지를 생성

<br>

### 2.3 VGG19 모델


```python
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

vgg = VGG19(include_top= False, weights='imagenet')

for layer in vgg.layers:
    print(layer.name)
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
    80142336/80134624 [==============================] - 10s 0us/step
    input_1
    block1_conv1
    block1_conv2
    block1_pool
    block2_conv1
    block2_conv2
    block2_pool
    block3_conv1
    block3_conv2
    block3_conv3
    block3_conv4
    block3_pool
    block4_conv1
    block4_conv2
    block4_conv3
    block4_conv4
    block4_pool
    block5_conv1
    block5_conv2
    block5_conv3
    block5_conv4
    block5_pool


- VGG19 모델을 불러옴
- Dense 레이어를 제외한 나머지 특징 추출 레이어만 가져옴

<br>

### 2.4 특정 Layer 선택


```python
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

vgg.trainable = False
outputs = [vgg.get_layer(name).output for name in style_layers]
model = tf.keras.Model([vgg.input], outputs)
```

- VGG19에서 가져온 레이어는 학습되지 않도록 설정

<br>

### 2.5 Gram Matrix


```python
def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)
```

- 특징 추출값을 1차원의 벡터로 만듬
- 자기 자신의 전치행렬과 곱

<br>

### 2.6 텍스쳐에서 Gram Matrix를 계산


```python
style_image = plt.imread(style_path)
style_image = cv2.resize(style_image, dsize = (224, 224))
style_image = style_image / 255.0

style_batch = style_image.astype('float32')
style_batch = tf.expand_dims(style_batch, axis = 0)
style_output = model(preprocess_input(style_batch * 255.0))
```

- 이미지의 크기를 바꾸고 픽셀의 정규화
- Style Output : 특징 레이어를 통과한 후에 나타난 결과

<br>

### 2.7 특징 레이어들 중 하나


```python
print(style_output[0].shape)
plt.imshow(tf.squeeze(style_output[0][:, :, :, 0], 0), cmap='gray')
plt.show()
```

    (1, 224, 224, 64)


<img src = 'https://user-images.githubusercontent.com/60168331/97982006-7b333780-1e16-11eb-891c-b4926477c664.png'>


- 이렇게 생긴 특징 레이어

<br>

### 2.8 텍스쳐의 Gram Matrix의 분포


```python
style_outputs = [gram_matrix(out) for out in style_output]

plt.figure(figsize=(12, 10))

for c in range(5):
    plt.subplot(3, 2, c + 1)
    array = sorted(style_outputs[c].numpy()[0].tolist())
    array = array[::-1]
    plt.bar(range(style_outputs[c].shape[0]), array)
    plt.title(style_layers[c])
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/97982009-7bcbce00-1e16-11eb-96a5-fc43586f2920.png'>


- 5개의 레이어는 이런색의 분포를 가지고 있음

<br>

### 2.9 함수 생성


```python
def get_outputs(image):
    image_batch = tf.expand_dims(image, axis=0)
    output = model(preprocess_input(image_batch * 255.0))
    outputs = [gram_matrix(out) for out in output]
    return outputs


def get_loss(outputs, style_outputs):
    return tf.reduce_sum([tf.reduce_mean((o-s) ** 2) for o, s in zip(outputs, style_outputs)])


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
```

- 타겟 텍스쳐에서 Gram Matrix를 구하는 함수
- 원본 텍스쳐의 Gram Matrix값과 타겟 텍스쳐에서의 MSE를 구하는 함수

<br>

### 2.10 Gradient Tape


```python
opt = tf.optimizers.Adam(learning_rate=0.2, beta_1=0.99, epsilon=1e-1)

@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = get_outputs(image)
        loss = get_loss(outputs, style_outputs)
        
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
```

- 학습해야할 가중치가 존재하지 않음
- 존재하는 것은 2개의 이미지와 MSE
- Gradient Tape은 학습되지 않는 가중치(혹은 입력)에 대해 Gradient를 계산함

<br>

### 2.11 epoch 설정, 애니메이션 효과를 위한 준비


```python
import IPython.display as display
import time
import imageio

start = time.time()
image = tf.Variable(target_image)

epochs = 50
steps_per_epoch = 100
```

<br>

### 2.12 텍스쳐 합성


```python
step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
    if n % 5 == 0 or n == epochs - 1:
        imageio.imwrite('./data/NeuralStyleTransfer2/style_epoch_{0}.png'.format(n), image.read_value().numpy())
    display.clear_output(wait=True)
    plt.axis('off')
    plt.imshow(image.read_value())
    plt.title('Train step:{}'.format(step))
    plt.show()
end = time.time()
print('Total time : {:.1f}'.format(end-start))
```

<img src = 'https://user-images.githubusercontent.com/60168331/97982011-7bcbce00-1e16-11eb-929d-dc4c1aa3f18a.png'>


    Total time : 1294.9


- 점점 합성이 됨 (중간에 100 step 마다 사진이 출력됨)
- 시간이 좀 걸림.
- 그리고 사진이 거칠어 보인다.

<br>

### 2.13 결과물의 개선


```python
def high_pass_x_y(image):
    x_var = image[:, 1:, :] - image[:, :-1, :]
    y_var = image[1:, :, :] - image[:-1, :, :]
    return x_var, y_var


def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_mean(x_deltas ** 2) + tf.reduce_mean(y_deltas ** 2)
```

- variation loss : 인접한 픽셀과의 차이값, 이 차이가 작을 수록 매끄러워 보임
- high_pass_x_y : 인접한 픽셀과의 차이를 구하함수
- total_variation_loss : high_pass_x_y의 결과를 제곱해서 평균 후 더하는 RMS 함수 작성

<br>

### 2.14 Variation 상황


```python
print('Target : ', total_variation_loss(image.read_value()))
print('Noise : ', total_variation_loss(tf.random.uniform(style_image.shape)))
print('Original : ', total_variation_loss(style_image))
```

    Target :  tf.Tensor(0.102714084, shape=(), dtype=float32)
    Noise :  tf.Tensor(0.33238393, shape=(), dtype=float32)
    Original :  tf.Tensor(0.03641251305469577, shape=(), dtype=float64)


- Target, Noise, Original의 Variation 상황

<br>

### 2.15 Variation Loss를 전체 Loss 추가


```python
total_variation_weight = 1e9
style_weight = 1e-1

@tf.function()
def train_step(image):
    with tf.GradientTape () as tape:
        outputs = get_outputs(image)
        loss = style_weight * get_loss(outputs, style_outputs)
        loss += total_variation_weight * total_variation_loss(image)
        
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
```

<br>

### 2.16 재시도


```python
start = time.time()

target_image = tf.random.uniform(style_image.shape)
image = tf.Variable(target_image)

epochs = 50
steps_per_epoch = 100
```

<br>

### 2.17 시작


```python
step = 0

for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
    if n % 5 == 0 or n == epochs - 1:
        imageio.imwrite('./data/NeuralStyleTransfer2/Style_variation_epoch_{0}.png'.format(n), image.read_value().numpy())
    display.clear_output(wait = True)
    plt.imshow(image.read_value())
    plt.title('Train step: {}'.format(step))
    plt.show()
    
end = time.time()
print('Total time : {:.1f}'.format(end - start))
```

<img src = 'https://user-images.githubusercontent.com/60168331/97982012-7c646480-1e16-11eb-8647-858e901debe7.png'>


    Total time : 1487.6


- 아까전보다 더 매끄러운 결과가 나옴.
- 뭔가 이쁨.

<br>

### 2.18 Variation 결과


```python
print('Target : ', total_variation_loss(image.read_value()))
print('Original : ', total_variation_loss(style_image))
```

    Target :  tf.Tensor(0.029525734, shape=(), dtype=float32)
    Original :  tf.Tensor(0.03641251305469577, shape=(), dtype=float64)


<br>

## 3. 신경 스타일 전이 실습
---
### 3.1 풍경 사진 Load


```python
content_path = tf.keras.utils.get_file('content.jpg', 'http://bit.ly/2mAfUX1')

content_image = plt.imread(content_path)
max_dim = 512
long_dim = max(content_image.shape[:-1])
scale = max_dim / long_dim
new_height = int(content_image.shape[0] * scale)
new_width = int(content_image.shape[1] * scale)

content_image = cv2.resize(content_image, dsize=(new_width, new_height))
content_image = content_image / 255.0
plt.figure(figsize=(8, 8))
plt.imshow(content_image)
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/97982014-7cfcfb00-1e16-11eb-9563-d615d1d64f83.png'>


- 실습용 사진을 가져옴
- 이 사진에 앞에서 만든 신경스타일을 전이 시킬것

<br>

### 3.2 Content 특징 추출 모델


```python
content_batch = content_image.astype('float32')
content_batch = tf.expand_dims(content_batch, axis = 0)

content_layers = ['block5_conv2']

vgg.trainalble = False
outputs = [vgg.get_layer(name).output for name in content_layers]
model_content = tf.keras.Model([vgg.input], outputs)
content_output = model_content(preprocess_input(content_batch * 255.0))
```

<br>

### 3.3 Content Output과 Loss의 정의


```python
def get_content_output(image):
    image_batch = tf.expand_dims(image, axis = 0)
    output = model_content(preprocess_input(image_batch * 255.0))
    return output


def get_content_loss(image, content_output):
    return tf.reduce_sum(tf.reduce_mean(image-content_output) ** 2)
```

<br>

### 3.4 Optimizer와 설정값 선정


```python
opt = tf.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.99, epsilon = 1e-1)

total_variation_weight = 1e9
style_weight = 1e-2
content_weight = 1e4
```

<br>

### 3.5 전체 Loss에 Content Loss 추가


```python
@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = get_outputs(image)
        output2 = get_content_output(image)
        loss = style_weight * get_loss(outputs, style_outputs)
        loss += content_weight * get_content_loss(output2, content_output)
        loss += total_variation_weight * total_variation_loss(image)
        
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
```

<br>

### 3.6 Fit 


```python
start = time.time()

image = tf.Variable(content_image.astype('float32'))

epochs = 20
stpes_per_epoch = 100

step = 0

for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print('.', end='')
    if n % 5 == 0 or n == epochs - 1:
        imageio.imwrite('./data/NeuralStyleTransfer2/Style_{0}_content_{1}_transfer_epoch_{2}.png'.format(
            style_weight, content_weight, n), image.read_value().numpy())
    display.clear_output(wait=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(image.read_value())
    plt.title(f'Train Step : {step}')
    plt.show()

end = time.time()
print('Total Time : {:.1f}'.format(end-start))
```

<img src = 'https://user-images.githubusercontent.com/60168331/97982018-7e2e2800-1e16-11eb-9edb-ad3c90d55a87.png'>


    Total Time : 2435.1


- 꽤 그럴싸해 보인다.
- 다만 시간이 엄청 오래걸림..

<br>

## 4. 요약
---
### 4.1 요약
- 그림의 스타일(특징)을 파악해서, 사진을 해당 그림처럼 만들어주는 내용을 해보았다.
- 엄청 신기했고, 유명화가의 그림 스타일을 파악해서, 사진을 넣고 바꾸는것도 가능할듯 하다.
- 해당 사진들은 <https://github.com/hmkim312/datas/tree/main/NeuralStyleTransfer2>{:target="_blank"} 에 있음.