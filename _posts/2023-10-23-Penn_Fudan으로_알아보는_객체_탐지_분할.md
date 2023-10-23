---
title: Penn-Fudan으로 알아보는 객체 탐지(Object Detection), 분할(Segmentation) with FasterRCNN.md
author: HyunMin Kim
date: 2023-10-23 00:00:00 0000
categories: [Data Science, CV]
tags: [Object Detection, Segmentation]
image: /assets/img/post/2023-10-23/02.png
---

해당 게시물은 [Torch Vision의 객체 감지 미세조정 튜토리얼](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)을 참고하여 작성되었습니다.
Pytorch에서 제공하는 Coco 데이터로 사전 훈련된 FasterRCNN을 활용하여 보행자 감지(detection) 및 분할(segmentation)을 위해 Penn-Fudan 데이터로 파라미터 튜닝을 진행합니다. Penn-Fudan 데이터는 345개의 보행자 정보가 포함된 총 170개의 이미지가 포함되어 있습니다.


## 1. Download
최초 1회 아래의 주석을 풀어 cutom function과 Penn-Fudan 데이터를 다운로드합니다.


```python
# !wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py
# !wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py
# !wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py
# !wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py
# !wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py
```


```python
# !wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
# !unzip PennFudanPed.zip
```

## 2. 패키지 import


```python
# Deafult
import os

# Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.patches as patches
import cv2
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

# Model
import torch
import torchvision
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# Custom Function
import utils
from engine import train_one_epoch, evaluate
```


```python
# 버전 확인
# Torchvision의 버전이 0.15이하이면 앞으로 진핼할 아래의 코드는 실행이 되지 않음
print('Torch Version : ',torch.__version__)
print('Torchvision Version : ',torchvision.__version__)
print('Matplotlib Version : ',matplotlib.__version__)
print('Cv2 Version : ',cv2.__version__)
```

    Torch Version :  2.1.0+cu118
    Torchvision Version :  0.16.0+cu118
    Matplotlib Version :  3.4.3
    Cv2 Version :  4.8.1


## 3. Penn-Fudan Dataset 확인
PennFudan Dataset은 345개의 보행자 정보와 170개의 이미지 파일로 구성되어있으며 Annotation파일에는 세그먼트, 바운딩 박스, 이미지, 마스크 파일 이름의 정보가 포함되어 있습니다.
```python
폴더 구조
PennFudanPed/
    PedMasks/
        FudanPed00001_mask.png
        FudanPed00002_mask.png
        FudanPed00003_mask.png
        FudanPed00004_mask.png
        ...
    PNGImages/
        FudanPed00001.png
        FudanPed00002.png
        FudanPed00003.png
        FudanPed00004.png
        ...
    Annotation/
        FudanPed00001.txt
        FudanPed00002.txt
        FudanPed00003.txt
        FudanPed00004.txt
        ...
        
```


```python
# 샘플 데이터 확인
def draw_images(annotation_file_path):
    # 어노테이션 파일 읽기
    with open(annotation_file_path, 'r') as file:
        lines = file.readlines()

    # 바운딩 박스 정보와 마스크 이미지 경로 추출
    bounding_boxes = []
    mask_image_path = None
    for line in lines:
        if line.startswith('Bounding box for object'):
            coordinates = line.split(': ')[1].strip().replace('(', '').replace(')', '').split(' - ')
            xmin, ymin = map(int, coordinates[0].split(', '))
            xmax, ymax = map(int, coordinates[1].split(', '))
            bounding_boxes.append(((xmin, ymin), (xmax, ymax)))
        elif line.startswith('Pixel mask for object'):
            mask_image_path = line.split(': ')[1].strip().replace('"', '')

    # 이미지와 마스크 이미지 불러오기
    image_path = annotation_file_path.replace('Annotation', 'PNGImages').replace('.txt', '.png')
    png_image = cv2.imread(image_path)
    png_image_rgb = cv2.cvtColor(png_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.imread(mask_image_path, 0)  # Grayscale

    # 이미지에 바운딩 박스 그리기
    boundingbox_image = png_image_rgb.copy()
    for box in bounding_boxes:
        cv2.rectangle(boundingbox_image, box[0], box[1], (255, 0, 0), 2)  # Red bounding box

    # 이미지 출력
    fig, ax = plt.subplots(1, 3, figsize=(8, 9))
    ax[0].imshow(png_image_rgb)
    ax[0].axis('off')  # Hide axes
    ax[0].set_title('Basic Image')
    ax[1].imshow(boundingbox_image)
    ax[1].axis('off')  # Hide axes
    ax[1].set_title('Bounding Image')
    ax[2].imshow(mask_image)
    ax[2].axis('off')  # Hide axes
    ax[2].set_title('Segmentation Image')
    plt.show()

for i in range(1,6):
    annotation_file_path = f'PennFudanPed/Annotation/FudanPed0000{i}.txt'
    draw_images(annotation_file_path)
```


<img src="/assets/img/post/2023-10-23/01.png" width=auto height=auto max-width=500>
    

    


## 4. 데이터 세트 정의
이미지의 Detection 및 Segmentation을 위해 torch의 dataset 클래스를 상속하여 Custom Dataset을 구성합니다.

1. Image
    - [3, H, W]의 텐서 shape 혹은 PIL Image의 크기 [H, W] (torchvision.tv_tensors.Image)

2. target
    - boxes : [N, 4]의 shape [x0, y0, x1, y1] (torchvision.tv_tensors.BoundingBoxes)
    - labels : 텐서 shape 정수 [N] (torch.Tensor)
    - image_id : 이미지를 식별하기 위한 고유 ID
    - area : BoundingBoxes의 영역 (torch.Tensor)
    - iscrowd : 텐서 shape의 uint8[N] (torch.Tensor)
    - masks : segmentation의 정보 [N, H, W] (torchvision.tv_tensors.Mask)


```python
class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # 모든 이미지 파일으 불러오고 정렬
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # 이미지와 마스크 불러오기
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)
        obj_ids = torch.unique(mask)
        # 첫번째 ID는 Background 이므로 제거 (CoCodataset 기준)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # 색상으로 인코딩된 마스크를 세트로 분할
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # 각 마스크의 Bounding box 좌표
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 모든 인스턴스가 iscrod라고 가정
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # 샘플과 대상을 torchvision tv_tensors로 래핑합니다.
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        # 이미지 transform 설정
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
```

## 5. 모델 정의
Coco dataset으로 사전 훈련된 모델을 기반으로 Penn-Fudan Dataset에 맞춰 Fine-Tuning을 진행할 FastRCNN 모델을 구성합니다.
Feature를 추출할 모델은 Resnet50을 사용했습니다. 또한, Segmentation을 하기 위해 MaskRCNN도 사용합니다.


```python
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    # Coco dataset으로 사전 훈련된 resnet50 불러오기
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # classification 모델의 input feature 갯수 가져오기
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 모델의 head 변경
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # segmentation 모델의 input feature 갯수 가져오기
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # 수정한 Layer들 model에 적용
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model
```

## 6. 모델 훈련
데이터셋을 데이터 로더에 넣고 모델을 Fine-Tuning 합니다.


```python
# 이미지 변환 함수
def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)
```


```python
# GPU or CPU 설정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 사람, Background class 설정
num_classes = 2

# 이미지 변환 및 Dataset 생성
dataset = PennFudanDataset('data/PennFudanPed', get_transform(train=True))
dataset_test = PennFudanDataset('data/PennFudanPed', get_transform(train=False))

# Train, Test Dataset 분할
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# 데이터 로더 정의
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn
)

# 모델 생성
model = get_model_instance_segmentation(num_classes)
model.to(device)

# Optimizer 구성
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# 스케쥴러 구성
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# 5회 학습
num_epochs = 5

for epoch in range(num_epochs):
    # Train - 10회 마다 loss 출력
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
    # learning rate 업데이트
    lr_scheduler.step()
    # Test 데이터로 평가
    evaluate(model, data_loader_test, device=device)
print("종료")
```

    Epoch: [0]  [ 0/60]  eta: 0:02:10  lr: 0.000090  loss: 3.3485 (3.3485)  loss_classifier: 0.8555 (0.8555)  loss_box_reg: 0.2014 (0.2014)  loss_mask: 2.2850 (2.2850)  loss_objectness: 0.0051 (0.0051)  loss_rpn_box_reg: 0.0016 (0.0016)  time: 2.1828  data: 0.0141  max mem: 1945
    Epoch: [0]  [59/60]  eta: 0:00:00  lr: 0.005000  loss: 0.3216 (0.7944)  loss_classifier: 0.0498 (0.1673)  loss_box_reg: 0.1535 (0.2185)  loss_mask: 0.1549 (0.3942)  loss_objectness: 0.0014 (0.0076)  loss_rpn_box_reg: 0.0055 (0.0068)  time: 0.2061  data: 0.0129  max mem: 2765
    Epoch: [0] Total time: 0:00:14 (0.2396 s / it)
    creating index...
    index created!
    Test:  [ 0/50]  eta: 0:00:05  model_time: 0.0931 (0.0931)  evaluator_time: 0.0033 (0.0033)  time: 0.1008  data: 0.0042  max mem: 2765
    Test:  [49/50]  eta: 0:00:00  model_time: 0.0385 (0.0639)  evaluator_time: 0.0025 (0.0043)  time: 0.0587  data: 0.0057  max mem: 2765
    Test: Total time: 0:00:03 (0.0746 s / it)
    Averaged stats: model_time: 0.0385 (0.0639)  evaluator_time: 0.0025 (0.0043)
    Accumulating evaluation results...
    DONE (t=0.01s).
    Accumulating evaluation results...
    DONE (t=0.01s).
    IoU metric: bbox
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.720
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.990
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.942
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.654
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.727
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.285
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.771
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.771
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.776
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.770
    IoU metric: segm
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.740
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.990
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.948
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.533
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.754
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.295
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.772
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.775
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.753
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.779
    
    ******Result******
    -----Bounding BOX AP IoU=0.50:0.95: 0.72-----
    -----Segmentation AP IoU=0.50:0.95: 0.74-----
    
    ......(생략)......
    
    Epoch: [4]  [ 0/60]  eta: 0:00:13  lr: 0.000500  loss: 0.2298 (0.2298)  loss_classifier: 0.0407 (0.0407)  loss_box_reg: 0.0676 (0.0676)  loss_mask: 0.1175 (0.1175)  loss_objectness: 0.0001 (0.0001)  loss_rpn_box_reg: 0.0040 (0.0040)  time: 0.2326  data: 0.0141  max mem: 3162
    Epoch: [4]  [59/60]  eta: 0:00:00  lr: 0.000500  loss: 0.1828 (0.1846)  loss_classifier: 0.0246 (0.0256)  loss_box_reg: 0.0360 (0.0391)  loss_mask: 0.1087 (0.1163)  loss_objectness: 0.0002 (0.0007)  loss_rpn_box_reg: 0.0020 (0.0028)  time: 0.1864  data: 0.0111  max mem: 3162
    Epoch: [4] Total time: 0:00:11 (0.1938 s / it)
    creating index...
    index created!
    Test:  [ 0/50]  eta: 0:00:02  model_time: 0.0391 (0.0391)  evaluator_time: 0.0024 (0.0024)  time: 0.0459  data: 0.0042  max mem: 3162
    Test:  [49/50]  eta: 0:00:00  model_time: 0.0395 (0.0399)  evaluator_time: 0.0017 (0.0026)  time: 0.0487  data: 0.0057  max mem: 3162
    Test: Total time: 0:00:02 (0.0489 s / it)
    Averaged stats: model_time: 0.0395 (0.0399)  evaluator_time: 0.0017 (0.0026)
    Accumulating evaluation results...
    DONE (t=0.00s).
    Accumulating evaluation results...
    DONE (t=0.00s).
    IoU metric: bbox
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.833
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.992
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.956
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.682
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.849
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.340
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.869
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.869
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.782
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.882
    IoU metric: segm
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.789
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.992
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.957
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.631
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.804
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.315
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.816
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.816
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.741
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.827
    
    ******Result******
    -----Bounding BOX AP IoU=0.50:0.95: 0.833-----
    -----Segmentation AP IoU=0.50:0.95: 0.789-----
    
    종료


5회 반복으로 학습한 결과 Object Detection의 IoU는 0.72에서 0.833까지 0.133상승했으며 segmentation은 0.74에서 0.789으로 0.049 상승된 수치를 보여주었습니다.

## 7. 예측 결과 확인
샘플 데이터를 넣고 Objectdetection과 segmentation의 결과를 확인해보니, 생각보다 결과가 잘 나온것을 알 수 있습니다.


```python
image = read_image("./PennFudanPed/PNGImages/PennPed00040.png")
eval_transform = get_transform(train=False)

# 모델 평가 모드 후 예측
model.eval()
with torch.no_grad():
    x = eval_transform(image)
    x = x[:3, ...].to(device)
    predictions = model([x, ])
    pred = predictions[0]


# 이미지 변환
image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
image = image[:3, ...]
pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
pred_boxes = pred["boxes"].long()
output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

# Mask 생성
masks = (pred["masks"] > 0.7).squeeze(1)
output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

# 이미지 확인
plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0))
plt.axis('off')
plt.show()
```

<img src="/assets/img/post/2023-10-23/02.png" width=auto height=auto max-width=500>