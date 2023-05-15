---
title: Pytorch를 사용하여 CIFAR10 이미지 분류기 만들기
author: HyunMin Kim
date: 2023-05-07 00:00:00 0000
categories: [Data Science, Deep Learning]
tags: [Deep Learning, CIFAR10]
---

## CIFAR10 이미지 분류기 만들기
- 합성곱 신경망(Convolution Neural Network)을 사용하여 성능이 향상된 이미지 분류기 생성 (w.Vgg16)

### 목차
1. 개요 및 결론 요약
1. CIFAR-10 데이터 불러와서 이미지 확인
1. Simple 합성곱 신경망 분류기 생성
1. HyperParameter 및 Loss Function 정의
1. Simple CNN Training 및 성능 확인
1. VGG16 Fine Tunning 및 성능 확인
1. 레퍼런스

## 1. 개요 및 결론 요약
### 1.1 개요
- CNN을 활용하여 직접 이미지 분류기를 만들어 성능을 확인하고, Pre-trainded된 모델을 Fine Tunning하여 성능을 비교하여 얼마나 차이나는지 확인함.
    - Simple Convolution Neural Network를 생성하여 CIFAR-10 이미지 데이터를 구별하는 분류기를 생성하여 성능을 확인함.
    - Pre Trained된 VGG16를 CIFAR-10 데이터로 Fine Tunning 후 성능을 직접 구축한 Simple CNN 대비 얼마나 성능이 좋아졌는지 확인함.
    - 두 모델은 모두 같은 하이퍼파라미터와 손실 함수를 사용하였음

### 1.2 결론
- Simple CNN 모델의 성능 58%, VGG16 모델의 성능은 80.15%으로 Simple CNN 대비 VGG16이 22.15%p 높은 성능을 보임
- 직접 CNN 모델을 구축하는것도 좋으나, 실제 업무에서는 Pre-trained된 모델을 가져와 Fine Tunning하여 사용하는게 성능도 좋고 시간도 절약될 것으로 파악됨



```python
# Package import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import random
```

### 2. CIFAR-10 데이터 불러와서 이미지 확인
#### 2.1 CIFAR-10 데이터란?
- CIFAR-10 데이터는 컴퓨터 비전 분야의 이미지 분류 작업에 널리 사용되는 벤치마크 데이터 세트로 60,000개의 32x32x3 컬러 이미지로 구성되어 있음.
- 총 10개의 비행기, 자동차, 새, 고양이, 사슴, 개, 개구리, 말, 배, 트럭 클래스로 이루어져있음
- 데이터 세트의 크기가 작고 단순하기 때문에 새로운 모델이나 기술을 신속하게 테스트하는 데 적합함
- 해당 과제에서는 학습 40000장, 검증 10000장, 테스트 10000장으로 진행함


```python
# Set the random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```


```python
transform = transforms.Compose(
    # numpy array -> Torch Tensor로 변환
    [transforms.ToTensor(),
     # RGB 채널에 0.5를 뺀 후 표준편차를 0.5로 나누어 이미지 정규화
     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
)
```


```python
# batch size 정의
batch_size = 128

# 학습 데이터 다운
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# 테스트 데이터 다운
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 학습, 검증 데이터로더 정의
trainset_size = int(0.8 * len(trainset))
valset_size = len(trainset) - trainset_size
trainset, valset = torch.utils.data.random_split(trainset, [trainset_size, valset_size])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

# 테스트 데이터로더 정의
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
```

    Files already downloaded and verified
    Files already downloaded and verified



```python
train_samples = len(trainset)
val_samples = len(valset)
test_samples = len(testset)
image_shape = testset.data.shape[1:]

print(f'Number of Train samples: {train_samples}')
print(f'Number of Val samples: {val_samples}')
print(f'Number of Test samples: {test_samples}')
print(f'Image shape: {image_shape}')
```

    Number of Train samples: 40000
    Number of Val samples: 10000
    Number of Test samples: 10000
    Image shape: (32, 32, 3)



```python
# 이미지 확인을 위한 함수 정의
def imshow(img):
    # 정규화 해제
    img = img / 2 + 0.5 
    npimg = img.numpy()
    # npimg 축 순서 변경
    plt.figure(figsize=(12,18))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 이미지 가져오기
dataiter = iter(trainloader)
images, labels = next(dataiter)

# 이미지 보여주기
# images 변수에 저장된 이미지들을 그리드 형태로 만들어 시각화하는 코드
imshow(torchvision.utils.make_grid(images))
```


    
![output_8_0](https://user-images.githubusercontent.com/60168331/236676695-6188377d-3d6e-4ffd-ac81-a264abba7cfe.png)
    


### 3. 간단한 합성곱 신경망 분류기 생성
- Conv, Maxpool, fc를 사용하여 간단한 합성곱 신경망 분류기를 정의함


```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### 4. HyperParameter 및 Loss Function 정의
- Simple CNN, VGG16 모두 같은 HyperParameter를 사용함
    - Loss : CrossEntropy
    - Optimizer : Adam
    - Learing_rate : 0.001
    - Epochs : 10
    - Batch size : 128


```python
# hyper Parameter
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```


```python
# Loss Function
mymodel = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mymodel.parameters(), lr=learning_rate)
```

### 5. Simple CNN Training 및 성능 확인
- Train Accuracy : 58.56%
- Test Accuracy : 58%
- 개, 사슴, 고양이는 다른 Class에 비해 잘 맞추지 못했음


```python
num_epochs = 10
for epoch in range(num_epochs): 
a    running_loss = 0.0

    # Simple CNN 학습
    mymodel.train()
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = mymodel(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0

    # Simple CNN 검증
    mymodel.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = mymodel(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Accuracy of the network on the Val images: {100 * correct / total}%')

print('Finished Training')
```

    Epoch [1/10], Step [100/313], Loss: 2.0294
    Epoch [1/10], Step [200/313], Loss: 1.7495
    Epoch [1/10], Step [300/313], Loss: 1.6427
    Epoch [1/10], Accuracy of the network on the Val images: 42.42%
    Epoch [2/10], Step [100/313], Loss: 1.5582
    Epoch [2/10], Step [200/313], Loss: 1.5240
    Epoch [2/10], Step [300/313], Loss: 1.4954
    Epoch [2/10], Accuracy of the network on the Val images: 47.71%
    Epoch [3/10], Step [100/313], Loss: 1.4331
    Epoch [3/10], Step [200/313], Loss: 1.4126
    Epoch [3/10], Step [300/313], Loss: 1.4159
    Epoch [3/10], Accuracy of the network on the Val images: 51.1%
    Epoch [4/10], Step [100/313], Loss: 1.3540
    Epoch [4/10], Step [200/313], Loss: 1.3555
    Epoch [4/10], Step [300/313], Loss: 1.3372
    Epoch [4/10], Accuracy of the network on the Val images: 52.52%
    Epoch [5/10], Step [100/313], Loss: 1.3030
    Epoch [5/10], Step [200/313], Loss: 1.2751
    Epoch [5/10], Step [300/313], Loss: 1.2644
    Epoch [5/10], Accuracy of the network on the Val images: 54.08%
    Epoch [6/10], Step [100/313], Loss: 1.2288
    Epoch [6/10], Step [200/313], Loss: 1.2306
    Epoch [6/10], Step [300/313], Loss: 1.2140
    Epoch [6/10], Accuracy of the network on the Val images: 56.11%
    Epoch [7/10], Step [100/313], Loss: 1.1530
    Epoch [7/10], Step [200/313], Loss: 1.1663
    Epoch [7/10], Step [300/313], Loss: 1.1702
    Epoch [7/10], Accuracy of the network on the Val images: 56.47%
    Epoch [8/10], Step [100/313], Loss: 1.1332
    Epoch [8/10], Step [200/313], Loss: 1.1087
    Epoch [8/10], Step [300/313], Loss: 1.1053
    Epoch [8/10], Accuracy of the network on the Val images: 58.27%
    Epoch [9/10], Step [100/313], Loss: 1.0781
    Epoch [9/10], Step [200/313], Loss: 1.0766
    Epoch [9/10], Step [300/313], Loss: 1.0693
    Epoch [9/10], Accuracy of the network on the Val images: 58.45%
    Epoch [10/10], Step [100/313], Loss: 1.0229
    Epoch [10/10], Step [200/313], Loss: 1.0472
    Epoch [10/10], Step [300/313], Loss: 1.0406
    Epoch [10/10], Accuracy of the network on the Val images: 58.56%
    Finished Training



```python
# 정확도 확인
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = mymodel(images)
        
        # 가장 높은 값를 갖는 분류(class)를 정답으로
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
```

    Accuracy of the network on the 10000 test images: 58 %



```python
# 각 분류(class)에 대한 예측값 계산을 위해 class명 정의
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = mymodel(images)
        _, predictions = torch.max(outputs, 1)
        # 각 분류별로 올바른 예측 수 저장
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# 각 분류별 정확도(accuracy)를 출력
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
```

    Accuracy for class: plane is 60.8 %
    Accuracy for class: car   is 60.4 %
    Accuracy for class: bird  is 57.8 %
    Accuracy for class: cat   is 36.4 %
    Accuracy for class: deer  is 46.7 %
    Accuracy for class: dog   is 43.1 %
    Accuracy for class: frog  is 78.3 %
    Accuracy for class: horse is 60.2 %
    Accuracy for class: ship  is 73.5 %
    Accuracy for class: truck is 68.4 %


### 6. VGG16 Fine Tunning 및 성능 확인
#### 6.1 VGG16 성능
- Train Accuracy : 80.79%
- Test Accuracy : 80.15%
- Class별 성능의 차이는 나지만, 전체적으로 Simple CNN보단 성능이 좋음.

#### 6.2 VGG16이란?
- 이미지 인식 작업에 일반적으로 사용되는 딥러닝 신경망 아키텍처로 옥스포드 대학에서 개발됨.
- 16은 네트워크의 레이어 수를 나타내며, VGG16은 13개의 컨볼루션 레이어와 3개의 완전 연결 레이어로 구성됨.
- VGG16은 이미지 인식, 객체 감지, 심지어 아트 스타일 트랜스퍼와 같은 다양한 컴퓨터 비전 애플리케이션에 널리 사용되고 있음.
- Pytorch에서 모델을 다운받아 손쉽게 Fine Tunning하여 사용 가능함


```python
# Load the pre-trained VGG16 model
vgg16 = torchvision.models.vgg16(pretrained=True)

# 마지막 Layer를 CIFAR-10 데이터에 맞게 재정의
num_classes = 10
vgg16.classifier[6] = torch.nn.Linear(vgg16.classifier[6].in_features, num_classes)
```

    /usr/local/lib/python3.9/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
      warnings.warn(
    /usr/local/lib/python3.9/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
      warnings.warn(msg)



```python
# HypterParameter 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg16.parameters(), lr=learning_rate)

# VGG16 Fine Tunning
vgg16.to(device)
for epoch in range(num_epochs):
    running_loss = 0.0
    vgg16.train()
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = vgg16(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0
            
    # VGG16 검증
    vgg16.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = vgg16(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the Val images: {100 * correct / total}%')

print('Finished Training')
```

    Epoch [1/10], Step [100/313], Loss: 2.3092
    Epoch [1/10], Step [200/313], Loss: 1.8725
    Epoch [1/10], Step [300/313], Loss: 1.7153
    Accuracy of the network on the Val images: 34.89%
    Epoch [2/10], Step [100/313], Loss: 1.4934
    Epoch [2/10], Step [200/313], Loss: 1.3086
    Epoch [2/10], Step [300/313], Loss: 1.1836
    Accuracy of the network on the Val images: 59.82%
    Epoch [3/10], Step [100/313], Loss: 1.0705
    Epoch [3/10], Step [200/313], Loss: 0.9924
    Epoch [3/10], Step [300/313], Loss: 0.9499
    Accuracy of the network on the Val images: 67.57%
    Epoch [4/10], Step [100/313], Loss: 0.8214
    Epoch [4/10], Step [200/313], Loss: 0.7617
    Epoch [4/10], Step [300/313], Loss: 0.7767
    Accuracy of the network on the Val images: 75.64%
    Epoch [5/10], Step [100/313], Loss: 0.6590
    Epoch [5/10], Step [200/313], Loss: 0.6808
    Epoch [5/10], Step [300/313], Loss: 0.6602
    Accuracy of the network on the Val images: 77.98%
    Epoch [6/10], Step [100/313], Loss: 0.5766
    Epoch [6/10], Step [200/313], Loss: 0.5743
    Epoch [6/10], Step [300/313], Loss: 0.6239
    Accuracy of the network on the Val images: 78.29%
    Epoch [7/10], Step [100/313], Loss: 0.5245
    Epoch [7/10], Step [200/313], Loss: 0.5322
    Epoch [7/10], Step [300/313], Loss: 0.5020
    Accuracy of the network on the Val images: 80.36%
    Epoch [8/10], Step [100/313], Loss: 0.3884
    Epoch [8/10], Step [200/313], Loss: 0.4778
    Epoch [8/10], Step [300/313], Loss: 0.4630
    Accuracy of the network on the Val images: 77.58%
    Epoch [9/10], Step [100/313], Loss: 0.3769
    Epoch [9/10], Step [200/313], Loss: 0.4439
    Epoch [9/10], Step [300/313], Loss: 0.4386
    Accuracy of the network on the Val images: 78.98%
    Epoch [10/10], Step [100/313], Loss: 0.3738
    Epoch [10/10], Step [200/313], Loss: 0.3877
    Epoch [10/10], Step [300/313], Loss: 0.4045
    Accuracy of the network on the Val images: 80.79%
    Finished Training



```python
# Evaluate the model
vgg16.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = vgg16(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct / total}%')
```

    Accuracy of the network on the test images: 80.15%



```python
# 각 분류(class)에 대한 예측값 계산을 위해 class명 정의
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = vgg16(images)
        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# 각 분류별 정확도(accuracy)를 출력합니다
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
```

    Accuracy for class: plane is 84.7 %
    Accuracy for class: car   is 83.4 %
    Accuracy for class: bird  is 68.9 %
    Accuracy for class: cat   is 66.3 %
    Accuracy for class: deer  is 88.8 %
    Accuracy for class: dog   is 68.7 %
    Accuracy for class: frog  is 85.0 %
    Accuracy for class: horse is 83.6 %
    Accuracy for class: ship  is 89.1 %
    Accuracy for class: truck is 83.0 %


### 7. Reference
- https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html
