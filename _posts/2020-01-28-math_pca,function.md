---
title: PCA와 함수
author: HyunMin Kim
date: 2020-01-28 00:00:00 0000
categories: [Datascience, Math]
tags: [PCA, Functionc, Relu, Soft Max]
---

## 1. PCA와 함수
### 1.1 PCA
- 주성분 분석, 차원축소

- 잠재변수를 찾는것
	- 잠재변수란 측정되지는 않았지만 측정된 데이터의 기저에 숨어서 측정 데이터를 결정 짓는 변수
	- <img src="https://latex.codecogs.com/gif.latex?u_1%20%3D%20w_1x_1&plus;w_2x_2" /> 
### 1.2 PCA의 수학적 설명
- 데이터가 원점을 중심으로 존재하는 경우에는 벡터에 변환행렬을 곱하는 연산으로 투영벡터를 계산할 수 있다.
- <img src = "https://latex.codecogs.com/gif.latex?%5Chat%20x_i%20%3D%20Wx_i" />
- PCA의 목표는 변환 결과인 차원축소 벡터  <img src="https://latex.codecogs.com/gif.latex?%5Chat%20x_i" />의 벡터의 정보가 원래의 벡터 <img src="https://latex.codecogs.com/gif.latex?x_i" />가졌던 정보와 가장 유사하게 되는 변환행렬 <img src="https://latex.codecogs.com/gif.latex?W" />를 찾는 것이다.

- 목적함수 :  <img src="https://latex.codecogs.com/gif.latex?%5Carg%5Cmin_%7BW%7D%20%7C%7C%20X%20-%20X%20W%5E%7BT%7D%20W%20%7C%7C%5E2" />
  
### 1.3 사이킷런의 PCA 기능
```python
from sklearn.decomposition import PCA

pca1 = PCA(n_components=1)
X_low = pca1.fit_transform(X)
X2 = pca1.inverse_transform(X_low)

```
- 입력 인수
	- n_components : 정수
- 메서드
	- fit_transform() : 특징행렬을 낮은 차원의 근사행렬로 변환
	- inverse_transform() : 변환된 근사행렬을 원래의 차원으로 복귀
- 속성
	- mean_ : 평균
	- components_ : 주성분 벡터

## 2. 함수 
### 2.1 함수란
- 함수 : 입력값을 출력값으로 바꾸어 출력하는 관계 (Ex 자판기)
- 정의역 : 입력변수가 가질 수 있는 값의 집합
- 공역 : 출력변수가 가질수 있는 값의 집합

### 2.2 변수란
- 변수 : 어떤 숫자를 대표하는 기호
- 입력변수 : 입력값을 대표하는 변수
- 출력변수 : 출력값을 대표하는 변수
- <img src="https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%202x" />
- <img src="https://latex.codecogs.com/gif.latex?y=2x" />
	
### 2.3 연속과 불연속
- 불연속 : 함수의 값이 중간에 갑자기 변하는 것
- 연속 :  <img src="https://latex.codecogs.com/gif.latex?x" />가 조금 바뀌면  <img src="https://latex.codecogs.com/gif.latex?y" />도 조금 바뀌는 것

### 2.4 부호함수
- 입력이 양수이면 1, 음수이면 -1, 0이면 0을 출력하는 <img src="https://latex.codecogs.com/gif.latex?x%20%3D%200" />에서 불연속 함수, 넘파이에서 sign()으로 사용

-   <img src="https://latex.codecogs.com/gif.latex?%5Ctext%7Bsgn%7D%28x%29%3D%5Cbegin%7Bcases%7D%201%2C%20%26%20x%20%3E%200%2C%20%5C%5C%200%2C%20%26%20x%20%3D%200%2C%20%5C%5C%20-1%2C%20%26%20x%20%3C%200%5Cend%7Bcases%7D" />

### 2.5 단위계단함수
- 입력이 0 이상이면 1을, 0 미만이면 0을 출력하는 함수

- <img src="https://latex.codecogs.com/gif.latex?H%28x%29%20%3D%5Cbegin%7Bcases%7D%201%2C%20%26%20x%20%5Cge%200%2C%20%5C%5C0%2C%20%26%20x%20%3C%200%5Cend%7Bcases%7D" />


### 2.6 지시함수
- 함수 이름에 아래 첨자로 미리 지정된 값이 들어오면 출력이 1이되고 아니면 출력이 0이 되는 함수
- <img src="https://latex.codecogs.com/gif.latex?%5Cmathbb%7BI%7D_i%28x%29%3D%5Cdelta_%7Bix%7D%20%3D%5Cbegin%7Bcases%7D1%20%26%20%5Ctext%7B%20if%20%7D%20x%20%3D%20i%20%5C%5C%200%20%26%20%5Ctext%7B%20if%20%7D%20x%20%5Cneq%20i%20%5C%5C%20%5Cend%7Bcases%7D" />

### 2.7 역함수
- 어떤 함수의 입력/출력 관계와 정반대의 입출력 관계를 가지는 함수
- 역함수는 역수와 기호의 의미가 다름
- 역함수는 항상 존재하는것은 아니다.
- <img src="https://latex.codecogs.com/gif.latex?y%20%3D%20f%28x%29%2C%20%5C%3B%5C%3B%5C%3B%20%5Crightarrow%20%5C%3B%5C%3B%5C%3B%20x%20%3D%20f%5E%7B-1%7D%28y%29" />
- <img src="https://latex.codecogs.com/gif.latex?f%5E%7B-1%7D%28x%29%20%5Cneq%20f%28x%29%5E%7B-1%7D%20%3D%20%5Cdfrac%7B1%7D%7Bf%28x%29%7D" />

### 2.8 함수의 그래프
- 함수의 시각화는 그래프와 플롯을 사용한다

### 2.9 역함수의 그래프
- 원래의 함수에서 <img src="https://latex.codecogs.com/gif.latex?x" />축과 <img src="https://latex.codecogs.com/gif.latex?y" />이 바뀐 것이므로 <img src="https://latex.codecogs.com/gif.latex?y%20%3D%20x" />가 나타나는 직선(원점을 통과하는 기울기 1인 직선)을 대칭축으로 대칭인 함수의 그래프와 같다

### 2.10 다항식 함수
- 상수항 <img src="https://latex.codecogs.com/gif.latex?c_0" />, 일차항 <img src="https://latex.codecogs.com/gif.latex?c_1" />, 이차항 <img src="https://latex.codecogs.com/gif.latex?c_2x%5E2%2C%5Ccdots" />등의 거듭제곱 항의 선형 조합으로 이루어진 함수다.
- <img src="https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20c_0%20&plus;%20c_1x&plus;c_2x%5E2&plus;%20%5Ccdots&plus;%20c_nx%5En" />

### 2.11 최대 함수
- 두 인수중 큰값을 출력하는 함수
-  <img src="https://latex.codecogs.com/gif.latex?%5Cmax%28x%2C%20y%29%20%3D%20%5Cbegin%7Bcases%7D%20x%20%26%20%5Ctext%7B%20if%20%7D%20x%20%5Cgeq%20y%20%5C%5C%20y%20%26%20%5Ctext%7B%20if%20%7D%20x%20%3C%20y%20%5Cend%7Bcases%7D" />

### 2.12 최소 함수
- 두 인수중 작은값을 출력하는 함수
-	<img src="https://latex.codecogs.com/gif.latex?%5Cmin%28x%2C%20y%29%20%3D%20%5Cbegin%7Bcases%7D%20x%20%26%20%5Ctext%7B%20if%20%7D%20x%20%5Cleq%20y%20%5C%5C%20y%20%26%20%5Ctext%7B%20if%20%7D%20x%20%3E%20y%20%5Cend%7Bcases%7D" />

### 2.13 **ReLU** 함수
-  <img src="https://latex.codecogs.com/gif.latex?y%20%3D%200" />으로 고정하여 입력값  <img src="https://latex.codecogs.com/gif.latex?x" />가 양수이면 그대로 출력하고 음수일때는 0으로 만들때 사용한다.
- 해당 함수는 인공신경망(딥러닝)에서 쓰인다.
-  <img src="https://latex.codecogs.com/gif.latex?%5Cmax%28x%2C%200%29%20%3D%20%5Cbegin%7Bcases%7D%20x%20%26%20%5Ctext%7B%20if%20%7D%20x%20%5Cgeq%200%20%5C%5C%200%20%26%20%5Ctext%7B%20if%20%7D%20x%20%3C%200%20%5Cend%7Bcases%7D" />

### 2.14 지수함수
- 밑을 오일러 수  <img src="https://latex.codecogs.com/gif.latex?e" />(약 2.718)로 하여 거듭 제곱하는 함수
-  <img src="https://latex.codecogs.com/gif.latex?y%20%3D%20e%5Ex" />
- 지수함수는  <img src="https://latex.codecogs.com/gif.latex?y%20%3D%20exp%28x%29%20%3Dexpx"/>로 표시하기도 한다.
- 양수(<img src="https://latex.codecogs.com/gif.latex?e" />)를 거듭제곱한 값이므로 항상 양수다.
-  <img src="https://latex.codecogs.com/gif.latex?x%20%3D%200" />일 때  <img src="https://latex.codecogs.com/gif.latex?1" />이 된다.
-  <img src="https://latex.codecogs.com/gif.latex?x" />가 양의 무한대로 가면( <img src="https://latex.codecogs.com/gif.latex?x%20%5Crightarrow%20%5Cinfty" />), 양의 무한대로 다가간다.
-  <img src="https://latex.codecogs.com/gif.latex?x" />가 음의 무한대로 가면( <img src="https://latex.codecogs.com/gif.latex?x%20%5Crightarrow%20-%5Cinfty" />), 0으로 다가간다.
-  <img src="https://latex.codecogs.com/gif.latex?x_1%20%3E%20x_2" />이면  <img src="https://latex.codecogs.com/gif.latex?%5Cexp%7Bx_1%7D%20%3E%20%5Cexp%7Bx_2%7D" />이다.

### 2.15 로지스틱함수
- 지수함수의 변형 함수
- 항상 0과 1사이의 값이 나오며, 단조증가함
-  <img src="https://latex.codecogs.com/gif.latex?%5Csigma%28x%29%20%3D%20%5Cdfrac%7B1%7D%7B1%20&plus;%20%5Cexp%28-x%29%7D" />


### 2.16 로그함수
-  <img src="https://latex.codecogs.com/gif.latex?e" />를 거듭제곱하여 특정한 수  <img src="https://latex.codecogs.com/gif.latex?a" />가 되도록 하는 수를  <img src="https://latex.codecogs.com/gif.latex?%5Clog%20a" />라 표기하고 로그라고 읽는다
-  <img src="https://latex.codecogs.com/gif.latex?y%20%3D%20%5Clog%20x" />
-  <img src="https://latex.codecogs.com/gif.latex?x" />값, 즉 입력변수값이 양수이어야 한다. 0이거나 음수는 안된다.
-  <img src="https://latex.codecogs.com/gif.latex?x%20%3E%201" />면  <img src="https://latex.codecogs.com/gif.latex?y%20%3E%201" /> (양수)
-  <img src="https://latex.codecogs.com/gif.latex?x%3D1" />이면  <img src="https://latex.codecogs.com/gif.latex?y%3D0" />
-  <img src="https://latex.codecogs.com/gif.latex?0%20%3C%20x%20%3C1" />면  <img src="https://latex.codecogs.com/gif.latex?y%3C0" /> (음수)
-  <img src="https://latex.codecogs.com/gif.latex?x_1%20%3E%20x_2" />면  <img src="https://latex.codecogs.com/gif.latex?%5Clog%20x_1%20%3E%20%5Clog%20x_2" />이다. 
- 로그함수의 그래프는 지수함수의 역함수이다.
- 로그함수는 곱하기를 더하기로 변환한다
-  <img src="https://latex.codecogs.com/gif.latex?%5Clog%28x_1%20%5Ccdot%20x_2%29%20%3D%20%5Clog%20x_1%20&plus;%20%5Clog%20x_2" />
-  <img src="https://latex.codecogs.com/gif.latex?%5Clog%7B%5Cleft%28%5Cprod_i%20x_i%5Cright%29%7D%20%3D%20%5Csum_i%20%5Cleft%28%5Clog%7Bx_i%7D%5Cright%29" />
-  <img src="https://latex.codecogs.com/gif.latex?%5Clog%20x%5En%20%3D%20n%5Clog%20x" />
- 어떤함수에 로그를 적용해도 함수의 최고점, 최저점의 위치는 변하지 않는다.
- 로그함수는 0부터 1사이의 작은 값을 확대시켜 보여준다.
	- 사람관점 : 차이가 확대되어 구별을 잘 할수있게 해줌
	- 컴퓨터관점 : 부동소수점의 단점을 보완 0~1을 더 크게 보여주기때문에

### 2.17 소프트플러스함수
- 지수함수와 로그함수를 결합하여 만든 함수
- 0을 인수로 갖는 최대 함수와 비슷하지만,  <img src="https://latex.codecogs.com/gif.latex?x%20%3D%200" />근처에서 값이 부드럽게 변함
-  <img src="https://latex.codecogs.com/gif.latex?%5Czeta%28x%29%20%3D%20%5Clog%28%201%20&plus;%20%5Cexp%28x%29%29" />$$$

### 2.18 다변수함수
- 복수의 입력변수를 가지는 함수
-  <img src="https://latex.codecogs.com/gif.latex?z%3Df%28x%2Cy%29" />
- 3차원은 서피스 플롯, 컨투어 플롯으로 시각화

### 2.19 분리가능 다변수함수
- 다변수 함수를 단변수함수의 곱으로 표현
-  <img src="https://latex.codecogs.com/gif.latex?f%28x%2Cy%29%20%3D%20f_1%28x%29f_2%28y%29" />

### 2.20 다변수 다출력 함수
- 입력변수와 출력변수가 여러개인 함수

### 2.21 소프트 맥스함수(다변수 다출력 함수)
<img src="https://latex.codecogs.com/gif.latex?y%20%3D%20%5Cbegin%7Bbmatrix%7D%20y_1%20%5C%5C%20y_2%20%5C%5C%20y_3%20%5Cend%7Bbmatrix%7D%20%3DS%28x_1%2C%20x_2%2C%20x_3%29%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cdfrac%7B%5Cexp%28w_1x_1%29%7D%7B%5Cexp%28w_1x_1%29%20&plus;%20%5Cexp%28w_2x_2%29%20&plus;%20%5Cexp%28w_3x_3%29%7D%20%5C%5C%20%5Cdfrac%7B%5Cexp%28w_2x_2%29%7D%7B%5Cexp%28w_1x_1%29%20&plus;%20%5Cexp%28w_2x_2%29%20&plus;%20%5Cexp%28w_3x_3%29%7D%20%5C%5C%20%5Cdfrac%7B%5Cexp%28w_3x_3%29%7D%7B%5Cexp%28w_1x_1%29%20&plus;%20%5Cexp%28w_2x_2%29%20&plus;%20%5Cexp%28w_3x_3%29%7D%20%5C%5C%20%5Cend%7Bbmatrix%7D" />

- 모든 출력 원소는 0과 1사이값을 갖는다.
- 모든 출력 원소의 합은 1이다.
- 입력원소의 크기 순서와 출력원소의 크기 순서가 같다.
- 다변수 입력을 확률처럼 보이게 출력 한다.

 ### 2.22 함수의 평행이동
-  <img src="https://latex.codecogs.com/gif.latex?f%28x%29%20%5C%3B%5C%3B%20%5Crightarrow%20%5C%3B%5C%3B%20f%28x-a%29" />
-  <img src="https://latex.codecogs.com/gif.latex?-" />은 오른쪽으로 이동,  <img src="https://latex.codecogs.com/gif.latex?&plus;" />는 왼쪽으로 이동
-  <img src="https://latex.codecogs.com/gif.latex?f%28x%29%20%5C%3B%5C%3B%20%5Crightarrow%20%5C%3B%5C%3B%20f%28x%29&plus;b" />
-  <img src="https://latex.codecogs.com/gif.latex?-" />은 아래로 이동,  <img src="https://latex.codecogs.com/gif.latex?&plus;" />는 위로 이동

### 2.23 함수의 스케일링
- 단변수함수를 x축 방향으로 a배만큼 늘릴려면 함수를 다음처럼 변형한다.
<img src="https://latex.codecogs.com/gif.latex?f%28x%29%20%5C%3B%5C%3B%20%5Crightarrow%20%5C%3B%5C%3B%20f%5Cleft%28%5Cfrac%7Bx%7D%7Ba%7D%5Cright%29" /> 양변에서 늘린것 같은 모양
- 단변수함수를 y축 방향으로 b배만큼 늘릴려면 함수를 다음처럼 변형한다.
<img src="https://latex.codecogs.com/gif.latex?f%28x%29%20%5C%3B%5C%3B%20%5Crightarrow%20%5C%3B%5C%3B%20bf%28x%29" /> 위아래로 늘린것 같은 모양

