---
title: Sympy를 사용한 함수, 행렬의 미분과 적분
author: HyunMin Kim
date: 2020-01-30 00:00:00 0000
categories: [Datascience, Math]
tags: [Sympy, Gradient Vector, Derivative, Integral]
---

## 1. sympy를 사용한 함수 미분
### 1.1예측 모형의 성능
- 성능함수 : 모수를 결정하여 성능을 측정하는 함수 
- <img src = "https://latex.codecogs.com/gif.latex?%5Chat%7By%7D%20%3D%20w_1%20x_1%20&plus;%20w_2%20x_2%20&plus;%20%5Cldots%20&plus;%20w_N%20x_N%20%3D%20w%5ET%20x" /> 
- 손실함수 : 오차(e)가 가장 작아지는 함수
- 목적함수 : 최적화의 대상이 되는 모든 함수 (성능, 손실, 오차)
- 최적화 : 목적 함수를 가장 크거나 작게 만드는 함수
- 미분 : 입력값이 변했을때 출력값이 어떻게 변하는지 확인하는 행위

### 1.2 기울기
- x와 y의 증감에 대한 민감도
- <img src = "https://latex.codecogs.com/gif.latex?%5Clim_%7B%5CDelta%20x%20%5Crightarrow%200%7D%20%5Cdfrac%7Bf%28x%20&plus;%20%5CDelta%20x%29%20-%20f%28x%29%7D%7B%5CDelta%20x%7D" />

### 1.3 수치 미분
- 수치적으로 대략적인 기울기

```python
from scipy.misc import derivative

print(derivative(f, 0, dx=1e-6))
print(derivative(f, 1, dx=1e-6))
```

```
1.000000000001
-2.000000000002
```

- scipp.misc의 derivative() 로 사용

### 1.4 미분
- 어떤 함수로부터 그 함수 기울기를 출력하는 새로운 함수를 만들어내는 작업
- 도함수 : 미분으로 만들어진 함수

### 1.5 미분 가능
- 미분 가능 : 기울기를 구할수 있다.
- 미분 불가능 : 기울기를 구할수 없다

### 1.6 미분 공식
- 기본 미분공식
- 선형 조합법칙
- 곱셈 법칙
- 연쇄 법칙

### 1.7 기본 미분 공식
-  상수 : <img src = "https://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%28c%29%20%3D%200" />
-  거듭제곱 : <img src = "https://latex.codecogs.com/gif.latex?%28x%5En%29%20%3D%20n%20x%5E%7Bn-1%7D">
- 로그 : <img src ="https://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%28%5Clog%20x%29%20%3D%20%5Cdfrac%7B1%7D%7Bx%7D" />
- 지수: <img src = "https://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%28e%5Ex%29%20%3D%20e%5Ex"/>

### 1.8 선형 조합 법칙 
- 어떤 함수에 각각 상수를 곱한 후 더한 선형조합은 각 함수의 도함수를 선형조합 한것과 같다
- <img src = "https://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%5Cleft%28c_1%20f_1%20&plus;%20c_2%20f_2%20%5Cright%29%20%3D%20c_1%20%5Cdfrac%7Bdf_1%7D%7Bdx%7D%20&plus;%20c_2%20%5Cdfrac%7Bdf_2%7D%7Bdx%7D" />

### 1.9 곱셈 법칙
- 어떤 함수의 형태가 두 함수를 곱한것과 같을땐 다음과 같이 각 개별 함수의 도함수를 사용하여 원래 함수의 도함수를 구하는것
- <img src = "https://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%5Cbig%28%20f%20%5Ccdot%20g%20%5Cbig%29%20%3D%20f%20%5Ccdot%20%5Cdfrac%7Bdg%7D%7Bdx%7D%20&plus;%20%5Cdfrac%7Bdf%7D%7Bdx%7D%20%5Ccdot%20g" />

### 1.10 연쇄 법칙
- 미분하고자 하는 함수의 입력 변수가 다른 함수의 출력 변수인 경우 적용
- <img src = "https://latex.codecogs.com/gif.latex?%5Cdfrac%7Bdf%7D%7Bdx%7D%20%3D%20%5Cdfrac%7Bdh%7D%7Bdg%7D%20%5Ccdot%20%5Cdfrac%7Bdg%7D%7Bdx%7D"/>

### 1.11 2차 도함수
- 도함수를 한번 더 미분하여 만들어진 함수
- 2차 도함수의 값이 양수면 볼록하다
- 2차 도암수의 값이 음수면 오목하다

### 1.12 편미분
- 함수가 둘 이상의 독립변수를 가지는 다변수 함수인 경우에도 미분 즉, 기울기는 하나의 변수에 대해서만 구할 수 있다
- 편미분의 결과로 하나의 함수에 대해 여러 도함수가 나올 수 있다.
- 어떤 하나의 독립변수에 대해 미분할때는 다른 독립변수를 상수로 취급

### 1.13 다변수 함수의 연쇄 법칙
- 다변수 함수의 미분을 구할 때도 함수가 연결되어 이으면 연쇄 법칙이 적용된다.

### 1.14 2차 편미분
- 편미분에 대해 2차 도함수를 정의한것
- 슈와르츠 정리 : 함수가 연속이고 미분 가능하다면 미분의 순서는 상관없다

### 1.15 테일러 전개
- 함수의 기울기를 근사화 하는것

### 1.16. sympy

- symbolic 연산을 지원하는 파이썬 패키지

```python
x = sympy.symbols('x')
```

- symbols()로 변수를 지정, 여러개 동시에 가능

```python
# 함수 정의
f = x * sympy.exp(x)

# 함수 미분
sympy.diff(f)
```

- diff()로 미분

```python
sympy.simplify(sympy.diff(f))
```

- simplify()로 소인수 분해

## 2. 적분
### 2.1 부정적분
- 정확하게 미분과 반대되는 개념, 즉 만 미분
- 도함수 -> 함수를 도출해내는 작업

### 2.2  편미분의 부정적분
- 편미분을 한 도함수에서 원래의 함수를 찾는 작업

### 2.3 다차 도함수와 다중적분
- 미분을 여러번 한 결과로 나온 다차 도함수로부터 원래의 함수를 찾아내려면 여러번 적분을 하는 다중적분이 필요
- <img src = "https://latex.codecogs.com/gif.latex?%5Ciint%20f%28x%2C%20y%29%20dydx"/>

### 2.4 sympy를 이용한 부정적분

```python
import sympy

sympy.init_printing(use_latex='mathjax')

x = sympy.symbols('x')
f = x * sympy.exp(x) + sympy.exp(x)
sympy.integrate(f)
```

```python
x, y = sympy.symbols('x y')
f = 2 * x + y

sympy.integrate(f, x)
```

- symyp.integrate()로 부정적분함

### 2.5 정적분
- 독립변수x가 어떤 구간 [a,b]사이일때 그 구간에서 함수f(x)의 값과 수평선(x축)이 이루는 면적을 구하는 행위
- <img src="https://latex.codecogs.com/gif.latex?%5Cint_%7Ba%7D%5E%7Bb%7D%20f%28x%29%20dx" />

- 미적분의 기본정리 : 부정적분으로 구한 F(x)를 이용하여 정적분을 구하는것
-  <img src = "https://latex.codecogs.com/gif.latex?%5Cint_%7Ba%7D%5E%7Bb%7D%20f%28x%29%20dx%20%3D%20F%28b%29%20-%20F%28a%29"/>

-  정적분은 부정적분을 한뒤 미적분학의 기본정리를 사용하여 구하거나
원래 함수의 면적 부분을 실제로 자게 쪼개어 면적을 근시하게 구하는 수치적분으로 구한다.

### 2.6 수치 적분
- 함수를 아주 작은 구간으로 나누어 실제면적을 계산함으로써 정적분 값을 구하는 방법

### 2.7 다변수 정적분
- 입력변수가 2개인 2차원 함수 f(x,y)의 경우에는 정적분을 다양한 방법으로 정의한다. 두변수로 모두 적분하는 것은 2차원 평면에서 주어진 사각형 영역 아래의 부피를 구하는것과 같다.
- <img src = "https://latex.codecogs.com/gif.latex?%5Cint_%7By%3Dc%7D%5E%7By%3Dd%7D%20%5Cint_%7Bx%3Da%7D%5E%7Bx%3Db%7D%20f%28x%2C%20y%29%20dx%20dy"/>

### 2.8 다차원 함수의 단일 정적분
- 2차원 함수에서 하나의 변수만 진짜 변수로 보고 나머지 하나는 상수로 간주하는 경우
- <img src = "https://latex.codecogs.com/gif.latex?%5Cint_a%5Eb%20f%28x%2C%20y%29%20dx"/>

## 3. 행렬의 미분
### 3.1 행렬미분 : 행렬을 입력이나 출력으로 가지는 함수를 미분
1. 벡터 x -> 스칼라 f
2. 행렬 x -> 스칼라 f
3. 스칼라 x -> 벡터 f
4. 벡터 x -> 행렬 f
5. 벡터 x -> 벡터 f
6. 벡터 x -> 행렬 f

### 3.2 스칼라를 벡터로 미분하는 경우
- 그레디언트 벡터 : 스칼라를 벡터로 미분하는 경우 경과를 열벡터로 표시
<img src = "https://latex.codecogs.com/gif.latex?%5Cnabla%20f%20%3D%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20%7Bx%7D%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_1%7D%5C%5C%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_2%7D%5C%5C%20%5Cvdots%5C%5C%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_N%7D%5C%5C%20%5Cend%7Bbmatrix%7D"/>
- 퀴버 플롯 : 컨투어 플롯 위에 그레디언트 벡터를 화살표로 나타낸 플롯
	- 그레디언트 벡터의 그기는 기울기를 의미하며, 벡터의 크기가 클수록 함수 곡면의 기울기가 커진다.
	- 그레디언트 벡터의 방향은 함수 곡면의 기울기가 가장 큰 방향, 즉 단위 길이당 함수값(높이)이 가장 크게 증가하는 방향을 가리킨다.
	- 그레디언트 벡터의 방향은 등고선 방향과 직교한다.

### 3.3 행렬 미분 법칙
- 행렬 미분 법칙 1 : 선형 모형
	- 선형 모형을 미분하면 그레디언트 벡터는 가중치다
	<img src ="https://latex.codecogs.com/gif.latex?%5Cnabla%20f%20%3D%20%5Cfrac%7B%5Cpartial%20%7Bw%7D%5E%7BT%7D%7Bx%7D%7D%7B%5Cpartial%20%7Bx%7D%7D%20%3D%20%5Cfrac%7B%5Cpartial%20%7Bx%7D%5E%7BT%7D%7Bw%7D%7D%7B%5Cpartial%20%7Bx%7D%7D%20%3D%20%7Bw%7D"/>

</br>

- 행렬 미분 법칙 2 : 이차 형식
	- 이차형식을 미분하면 행렬과 벡터의 곱으로 나타난다.
	- <img src = "https://latex.codecogs.com/gif.latex?%5Cnabla%20f%28x%29%20%3D%20%5Cfrac%7B%5Cpartial%20%7Bx%7D%5E%7BT%7D%7BA%7D%7Bx%7D%7D%7B%5Cpartial%20%7Bx%7D%7D%20%3D%20%28%7BA%7D%20&plus;%20%7BA%7D%5E%7BT%7D%29%7Bx%7D"/>


- 벡터를 스칼라로 미분하는 경우
	- 벡터를 스칼라로 미분하는 경우에는 결과를 행벡터로 표시한다.

- 벡터를 벡터로 미분하는 경우
    - 벡터를 벡터로 미분하면 결과로 나온 도함수는 2차원 배열 즉, 행렬이 된다.

- 행렬 미분 법칙 3 : 행렬과 벡터의 곱의 미분
    - 행렬 A와 벡터 x의 곱 Ax를 벡터x로 미분하면 A.T가 된다.
    <img src =  "https://latex.codecogs.com/gif.latex?%5Cnabla%20f%28x%29%20%3D%20%5Cdfrac%7B%5Cpartial%20%28%7BAx%7D%29%7D%7B%5Cpartial%20%7Bx%7D%7D%20%3D%20A%5ET" />
    - 자코비안행렬 : 함수의 출력변수와 입력변수가 모두 벡터(다차원)데이터인 경우에는 입력변수 각각과 출력변수 각각의 조합으로 만들어진 전치행렬
    - 헤시안행렬 : 다변수함수의 2차 도함수는 그레디언트 벡터를 입력변수로 미분한것


### 3.4 스칼라를 행렬로 미분
- 출력변수 f가 스칼라 값이고 입력변수 X가 행렬인 경우에는 도함수 행렬의 모양이 입력변수 행렬 x와 같다.

### 3.5 행렬 미분 법칙 4 : 행렬 곱의 대각성분
- 두 정방행렬을 곱해서 만들어진 행렬의 대각성분은 스칼라이며 이 스칼라를 뒤의 행렬로 미분하면 앞 행렬의 전치행렬이 나온다.
- <img src = "https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20X%7D%20%3D%20%5Cdfrac%7B%5Cpartial%20%5C%2C%20%5Ctext%7Btr%7D%20%28%7BW%7D%7BX%7D%29%7D%7B%5Cpartial%20%7BX%7D%7D%20%3D%20%7BW%7D%5ET"/>

### 3.6 행렬 미분 법칙 5 : 행렬식의 로그
- 행렬식은 스칼라값이고 이 값의 로그 값도 스칼라이며 이 값을 원래 행렬로 미분하면 원래 행렬의 역행렬의 전치 행렬이 된다.
- <img src = "https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20X%7D%20%3D%20%5Cdfrac%7B%5Cpartial%20%5Clog%20%7C%20%7BX%7D%20%7C%20%7D%7B%5Cpartial%20%7BX%7D%7D%20%3D%20%28%7BX%7D%5E%7B-1%7D%29%5ET"/>
  