---
title: 구글 colab과 vscode 연동하기
author: HyunMin Kim
date: 2022-01-23 00:00:00 0000
categories: [Data Science , ETC]
tags: [Vscode, Colab]
---

## 0. 들어가며
- 구글 코랩은 굉장히 가성비가 좋은 데이터분석툴이다.
- 쥬피터와 비슷한 환경을 가지고 있어 사용하는데 어색하지 않으며, GPU를 무료로 빌려주어 딥러닝 학습에도 많이 사용한다.
- vscode와 연동하여, 굳이 코랩으로 사용하지 않고 vscode 환경에서 사용하는 방법을 소개하려한다.

## 1. ngrok 설치
- https://dashboard.ngrok.com/get-started/setup
- 외부에서 로컬에 접속할수 있게 해주는 터널링 프로그램
- OS에 맞게 설치 (꼭 해야하는지는 확인 못함)
- https://dashboard.ngrok.com/get-started/your-authtoken
- 위의 경로에서 authtoken 생성 (유효 기간 8시간)
<img src="https://user-images.githubusercontent.com/60168331/150681204-20fa41f4-e79f-4482-bc69-ce37bd57c76b.png">


## 2. colab 설정
###  2.1 구글 drive와 colab 연동하기

```python
from google.colab import drive
drive.mount('/content/drive')
```
- google drive 마운트

```shell
!pip install colab-ssh --upgrade
```
```python
authtoken = 'ngrok에서 복사한 token'
password = '접속할때 쓸 password

from colab_ssh import launch_ssh
launch_ssh(authtoken, password)
```
- colab에 ssh launch 실행

<img src="https://user-images.githubusercontent.com/60168331/150681366-878af855-91f6-4b92-9558-3ef69ccf4815.png">

- host 정보 확인

## 3. vscode
- remote ssh 설치 (처음 1회)
- command + shift + p -> remote-ssh connect to host
- configure ssh hosts
- user/username/.ssh/config 선택
<img src="https://user-images.githubusercontent.com/60168331/150681410-df193f91-ac0a-4dd6-abe2-1a4f2cd31c18.png">

- colab host 정보 입력
<img src="https://user-images.githubusercontent.com/60168331/150681475-580d924e-4dd7-4b46-b6d6-e6a5b4af26f2.png">

- command + shift + p -> remote-ssh connect to host -> 방금 입력한 host 선택 

<img src="https://user-images.githubusercontent.com/60168331/150681519-3207609e-2355-4186-a1da-15c597c5ed1c.png">


- colab에서 설정한 비밀번호 입력후 사용
<img src='https://user-images.githubusercontent.com/60168331/150682787-b7733ab0-accb-4bcd-850f-46313dbc63f9.png'>

## 4. 결론
- ngrok를 사용하여 코랩과 vscode를 연동할수 있다.
- 또한 코랩의 gpu도 사용가능하다.
- 하지만 연동하고나면 jupyter 커널등을 다시 깔아줘야하는 등의 번거로움이 있다.
- 위의 번거로움 때문에 그냥 코랩을 쓸듯 하다.
- 애초에 코랩과 구글드라이브의 연동이 너무 편리하기 때문이다.