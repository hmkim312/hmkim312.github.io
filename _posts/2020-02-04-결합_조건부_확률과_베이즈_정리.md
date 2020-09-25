---
title: 결합, 조건부 확률과 베이즈 정리
author: HyunMin Kim
date: 2020-02-04 12:00:00 0000
categories: [Data Science, Math]
tags: [Probability, Joint Probability, Marginal Probability, Conditional Probability, Random Variable, Pgmpy, Bayesian Rule] 
---

## 1. 결합확률과 조건부 확률
---
### 1.1 결합확률(Joint Probabilit)
- 사건 A와 B가 동시에 발생할 확률
- *P(A,B)*

<br>

### 1.2 주변확률(Marginal Probability)
- 결합확률과 대비되는 개념으로 결합되지 않는 개별 사건의 확률
- *P(A) or P(B)*

<br>

### 1.3 조건부확률(Conditional Probability)
- B가 사실일 경우의 사건 A에 대한 확률을 **사건 B에 대한 사건 A의 조건부확률**이라고함
- _P(A \| B)_
- 사건 B가 진실이라면 사건 A에 대한 확률이 달라진다.

<br>

#### 1.3.1 조건부확률 *P(A | B)* 요약
- 사건 B가 발생한 경우의 사건 A의 확률
- 표본이 이벤트 B에 속한다는 새로운 **사실**을 알게 되었을때
- 이 표본이 사건 A에 속한다는 사실의 정확성 (**신뢰도**)이 어떻게 변하는지 알려줌

<br>

### 1.4 독립
- 사건 A와 사건B의 결합확률의 값이 다음과 같은 관계가 성립하면 **두 사건 A와 B는 서로독립** 이라고 정의함
- *P(A,B) = P(A)P(B)*
- 독립의 경우에는 조건부확률과 원래의 확률이 같아짐

<br>

### 1.5 원인과 결과, 근거와 추론, 가정과 조건부 결론
- 조건부 확률 *P(A \| B)*에서 사건(주장/명제) B,A는 각각
    - 가정(B)과 그 가정에 따른 조건부 결론(A)
    - 원인(B)과 결과(A)
    - 근거(B)와 추론(A)

- *P(A,B) = P(A \| B)P(B)*
    
<br>    

### 1.6 사슬 법칙
- 조건부확률과 결합확률의 관계를 확장한것
  - *P(X1,X2) = P(X1)P(X2 \| X1)*
  - *P(X1,X2,X3) = P(X3 \| X1,X2)P(X1,X2) = P(X1)P(X2 \| X1)P(X3 \| X1,X2)*
  - <img src = "https://latex.codecogs.com/gif.latex?P%28X_1%2C.....%2CX_N%29%20%3D%20P%28X_1%29%5CPi%5EN_%7BI%3D2%7DP%28X_I%7CX_1%2C....%2CX_%7Bi-1%7D%29"/>

<br>

### 1.7 확률변수(Random Variable)
- 확률적인 숫자 값을 출력하는 변수
- 결합확률의 확률분포는 각 확률변수가 가질 수 있는 값의 조합으로 표시

<br>

### 1.8 pgmpy 패키지를 사용한 결합확률과 조건부확률

```python
JointProbabilityDistribution(variables, cardinality, values)
```
- 결합확률 모형을 만드는데 쓰는 클래스
- variables: 확률변수의 이름 문자열의 리스트. 정의하려는 확률변수가 하나인 경우에도 리스트로 넣어야 한다.
- cardinality: 각 확률변수의 표본 혹은 배타적 사건의 수의 리스트
- values: 확률변수의 모든 표본(조합)에 대한 (결합)확률값의 리스트

<br>

```python
marginal_distribution(values, inplace = False)
```
- 인수로 받은 확률변수에 대한 주변확률분포를 구함
- values: 주변확률을 구할 확률변수의 이름 문자열 리스트
- inplace: True이면 객체 자신을 주변확률 모형으로 변화시킨다. False면 주변확률 모형 객체를 반환한다.

<br>

```python
marginalize(values, inplace=True)
```
-  인수로 받은 확률변수를 주변화하여 나머지 확률변수에 대한 주변확률분포를 구함
- values: 어떤 확률변수의 주변확률을 구하기 위해 없앨 확률변수의 이름 문자열 리스트
- inplace: True이면 객체 자신을 주변확률 모형으로 변화시킨다. False면 주변확률 모형 객체를 반환한다.

<br>

```python
conditional_distribution(values, inplace=True)
```

- 어떤 확률변수가 어떤 사건이 되는 조건에 대해 조건부확률값을 계산
- values: 주변확률을 구할 확률변수의 이름 문자열과 값을 묶은 튜플의 리스트
- inplace: True이면 객체 자신을 조건부확률 모형으로 변화시킨다. False면 조건부확률 모형 객체를 반환한다.

<br>

```python
check_independence()
```
- 두 확률변수 같의 독립을 확인

<br>

## 2. 베이즈 정리
---
### 2.1 베이즈 정리란 (Bayesian Rule)
- 조건부 확률을 구하는 공식
- <img src = "https://latex.codecogs.com/gif.latex?P%28A%7CB%29%20%3D%20%7BP%28B%7CA%29P%28A%29%20%5Cover%20P%28B%29%7D"/>
  - *P(A \| B)* : 사후확률. 사건 B가 발생한 후 갱신된 사건 A의 확률
  - *P(A)* : 사전확률. 사건 B가 발생하기 전에 가지고 있던 사건 A의 확률 
  - *P(B \| A)* : 가능도. 사건 A가 발생한 경우 사건B의 확률
  - *P(B)* : 정규화 상수 또는 증거, 확률의 크기 조정

- 사건 B가 발생함으로써 사건 A의 확률이 어떻게 변화하는지를 표현한 정리

<br>

### 2.2 베이즈 정리의 확장 1
- 서로 베타적 
  - <img src ="https://latex.codecogs.com/gif.latex?A_i%20%5Ccap%20A_j" />
- 완전(합집합이 표본공간) 
  - <img src ="https://latex.codecogs.com/gif.latex?A_1%20%5Ccup%20A_2%20%5Ccup%20%5Cdots%20%3D%20%5COmega"/>
  	
<br>

### 2.3  베이즈 정리의 확장의 증명
- <img src = "https://latex.codecogs.com/gif.latex?P%28A_1%7CB%29%20%3D%20%5Cdfrac%7BP%28B%7CA_1%29P%28A_1%29%7D%7BP%28B%29%7D%20%5C%5C%20%3D%20%5Cdfrac%7BP%28B%7CA_1%29P%28A_1%29%7D%7B%5Csum_i%20P%28A_i%2C%20B%29%7D%20%5C%5C%20%3D%20%5Cdfrac%7BP%28B%7CA_1%29P%28A_1%29%7D%7B%5Csum_i%20P%28B%7CA_i%29P%28A_i%29%7D"/>

<br>

### 2.4 검사시약문제
#### 2.4.1 사건
- 병에 걸리는 경우 : 사건 *D*
- 양성 반응을 보이는 경우 : 사건 *S*
- 병에 걸린 사람이 양성 반응을 보이는 경우 : 조건부 사건 *S \| D*
- 양성 반응을 보이는 사람이 병에 걸려 있을 경우 : 조건부 사건 *D \| S*
  
<br>

#### 2.4.2 문제
- *P(S \| D) = 0.99*가 주어졌을때, *P(D \| S)*를 구하라
  - <img src = "https://latex.codecogs.com/gif.latex?P%28D%7CS%29%20%3D%20%5Cdfrac%7BP%28S%7CD%29P%28D%29%7D%7BP%28S%29%7D"/>

- 이 병은 전체 인구 중 걸린 사람이 0.2%인 희귀병
    - *P(D) = 0.002*

- 이 병에 걸리지 않은 사람에게 시약 검사를 했을때, 양성 반응, 즉 잘못된 결과(False Positive)가 나타난 확률은 5%
    - <img src ="https://latex.codecogs.com/gif.latex?P%28S%7CD%5EC%29%20%3D%200.05"/>

<br>

### 2.4.3 문제 해결 및 결과
- <img src ="https://latex.codecogs.com/gif.latex?P%28D%7CS%29%20%3D%20%5Cdfrac%7BP%28S%7CD%29P%28D%29%7D%7BP%28S%29%7D%20%5C%5C%20%3D%20%5Cdfrac%7BP%28S%7CD%29P%28D%29%7D%7BP%28S%2CD%29%20&plus;%20P%28S%2CD%5EC%29%7D%20%5C%5C%20%3D%20%5Cdfrac%7BP%28S%7CD%29P%28D%29%7D%7BP%28S%7CD%29P%28D%29%20&plus;%20P%28S%7CD%5EC%29P%28D%5EC%29%7D%20%5C%5C%20%3D%20%5Cdfrac%7BP%28S%7CD%29P%28D%29%7D%7BP%28S%7CD%29P%28D%29%20&plus;%20P%28S%7CD%5EC%29%281-P%28D%29%29%7D%20%5C%5C%20%3D%20%5Cdfrac%7B0.99%20%5Ccdot%200.002%7D%7B0.99%20%5Ccdot%200.002%20&plus;%200.05%20%5Ccdot%20%281%20-%200.002%29%7D%20%5C%5C%20%3D%200.038"/>

- 시약반응에서 양성 반응을 보이는 사람이 실제로 병에 걸려 있을 확률은 약 3.8%이다.

<br>

### 2.5 Pgmpy를 사용한 베이즈 정리

- 피지엠파이 패키지는 베이즈 정리에 적용하는 BayesianModel 클래스를 제공함
- 베이즈 정리를 적용하려면 조건부확률을 구현하는 TabularCPD 클래스를 사용하여 사전확률과 가능도를 구하면 됨

```python
TabularCPD(variable, variable_card, value, evidence=None, evidence_card=None)
```

- variable: 확률변수의 이름 문자열
- variable_card: 확률변수가 가질 수 있는 경우의 수
- value: 조건부확률 배열. 하나의 열(column)이 동일 조건을 뜻하므로 하나의 열의 확률 합은 1이어야 한다.
- evidence: 조건이 되는 확률변수의 이름 문자열의 리스트
- evidence_card: 조건이 되는 확률변수가 가질 수 있는 경우의 수의 리스트
- evidence=None, evidence_card=None으로 인수를 주면 일반적인 확률도 구현가능
- 조건부확률을 구현하는 클래스(사전확률과 가능도를 구현하기 위한 선행작업)

<br>

```python
BayesianModel(variables)
```

- variables: 확률모형이 포함하는 확률변수 이름 문자열의 리스트
- BayesianModel 클래스는 다음 메서드를 지원함
  - add_cpds(): 조건부확률을 추가
  - check_model(): 모형이 정상적인지 확인. True면 정상적인 모형
- 확률변수들이 어떻게 결합되어 있는지 나타내는 확률 모형

<br>

```python
VarialbeElimination(model).query(variables, evidences)
```

- variables: 사후확률을 계산할 확률변수의 이름 리스트
- evidences: 조건이 되는 확률변수의 값을 나타내는 딕셔너리
- 사후확률을 계산함

<br>

### 2.6 베이즈 정리의 확장 2
- 베이즈정리에 추가적인 사건 *C*가 발생했을때의 베이즈정리
  - <img src = "https://latex.codecogs.com/gif.latex?P%28A%7CB%2CC%29%20%3D%20%5Cdfrac%7BP%28C%7CA%2CB%29P%28A%7CB%29%7D%7BP%28C%7CB%29%7D"/>

- 중복되는 조건 *B*를 삭제하여 외우면 쉽다
  - <img src = "https://latex.codecogs.com/gif.latex?P%28A%7CC%29%20%3D%20%5Cdfrac%7BP%28C%7CA%29P%28A%29%7D%7BP%28C%29%7D"/>

<br>

### 2.7 몬타 홀 문제
#### 2.7.1 문제 요약
- 세 문중에 하나를 선택하여 문뒤에 있는 **자동차**를 찾는 게임
- 세 문중에 하나를 선택하면 선택한 문을 제외한 다른 문을 열어주어 **염소**를 보여줌
- 위의 상황에서 선택했던 문을 바꾸는것이 유리한가?

<br>

#### 2.7.2 확률변수 설정
- 자동차가 있는 문을 나타내는 확률변수 *C*로 값은 0,1,2를 가질 수 있다.
- 참가자가 선택한 문을 나타내는 확률변수 *X*로 값은 0,1,2를 가질 수 있다.
- 진행자가 열어준 문을 나타내는 확률변수 *H*로 값은 0,1,2를 가질 수 있다.
- 참가자 선택 : <img src = "https://latex.codecogs.com/gif.latex?X_1"/>
- 진행자가 열어준 문 : <img src ="https://latex.codecogs.com/gif.latex?H_2"/>
  
<br>

#### 2.7.3 문제를 풀기 위한 핵심 사실(1)
- 자동차를 놓는 진행자는 참가자의 선택을 예측할 수 없고, 참가자는 자동차를 볼 수 없으므로 자동차의 위치와 참가자의 선택은 서로 독립적이다
- <img src = "https://latex.codecogs.com/gif.latex?P%28C%2CX%29%20%3D%20P%28C%29P%28X%29"/>

<br>

#### 2.7.4 문제를 풀기 위한 핵심 사실(2)
- 진행자가 어떤 문을 여는가가 자동차 위치 및 참가자 선택에 좌우 된다. 예를 들어 자동차가 0번 문 뒤에 있고 참가자가 1번 문을 서낵하면 진행자는 2번 문을 열어야 한다.
- <img src ="https://latex.codecogs.com/gif.latex?%5C%5CP%28H_0%7CC_0%2CX_1%29%20%3D%200%20%5C%5C%20P%28H_1%7CC_0%2CX_1%29%20%3D%200%5C%5C%20P%28H_2%7CC_0%2CX_1%29%20%3D%201%5C%5C"/>
  
<br>

#### 2.7.5 문제 해결 및 결과
- 자동차가 1번 문 뒤에 있는데, 참가자가 1번 문을 선택한 경우에는 0번 문과 2번문 둘다 열어도 된다. 따라서 진행자가 0번 문이나 2번 문을 열 확률은 0.5다

  - <img src = "https://latex.codecogs.com/gif.latex?%5C%5CP%28H_0%7CC_1%2CX_1%29%20%3D%20%5Cfrac12%20%5C%5CP%28H_1%7CC_1%2CX_1%29%20%3D%200%20%5C%5CP%28H_2%7CC_1%2CX_1%29%20%3D%20%5Cfrac12"/>
  
  
- 이 사실을 이용하여 참가자가 1번문을 선택하고 진행자가 2번문을 열어서 자동차가 없다는걸 보였다면 0번문 뒤에 차가 있을 확률은 다음과 같다.
  - <img src = "https://latex.codecogs.com/gif.latex?%5Cinline%20%5C%5CP%28C_0%5C%2C%7C%5C%2CX_1%2CH_2%29%20%3D%20%5Cdfrac%7BP%28C_0%2CX_1%2CH_2%29%7D%7BP%28X_1%2CH_2%29%7D%20%5C%5C%20%3D%20%5Cdfrac%7BP%28H_2%7CC_0%2CX_1%29P%28C_0%2CX_1%29%7D%7BP%28X_1%2CH_2%29%7D%20%5C%5C%20%3D%20%5Cdfrac%7BP%28C_0%29%7BP%28X_1%29%7D%7D%7BP%28H_2%7CX_1%29%7BP%28X_1%29%7D%7D%20%5C%5C%20%3D%20%5Cdfrac%7BP%28C_0%29%7D%7BP%28H_2%7CX_1%29%7D%20%5C%5C%20%3D%20%5Cdfrac%7BP%28C_0%29%7D%7BP%28H_2%2CC_0%7CX_1%29&plus;P%28H_2%2CC_1%7CX_1%29&plus;P%28H_2%2CC_2%7CX_1%29%7D%20%5C%5C%20%3D%20%5Cdfrac%7BP%28C_0%29%7D%7BP%28H_2%7CX_1%2CC_0%29P%28C_0%29&plus;P%28H_2%7CX_1%2CC_1%29P%28C_1%29&plus;P%28H_2%7CX_1%2CC_2%29P%28C_2%29%7D%20%5C%5C%20%3D%20%5Cdfrac%7B%5Cfrac13%7D%7B1%20%5Ccdot%20%5Cfrac13%20&plus;%20%5Cfrac12%5Ccdot%20%5Cfrac13%20&plus;%200%20%5Ccdot%20%5Cfrac13%7D%20%5C%5C%20%3D%20%5Cfrac23"/>
  
- 0번 문뒤에 자동차가 있을 확률이 2/3이기 때문에 보다 1번 문뒤에 자동차가 있을 확률(1/3)보다 2배 더 높다 (2/3 > 1/3)
