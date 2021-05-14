---
title: Requirements 사용법
author: HyunMin Kim
date: 2021-05-14 00:00:00 0000
categories: [Python, Virtual Environment]
tags: [Requirements, Pip,]
---

## Requirements 사용법
---
### 1. 사용 이유
- 테스트 서버에서 가상환경을 구축 후 개발 후 다른 서버에 배포할때, 테스트 서버에서 설치한 패키지나 라이브러리를 하나씩 설치하는것이 아닌 requirements를 제공하여 패키지 및 라이브러리를 손쉽게 설치하도록 하기 위해 사용한다.

<br>

### 2. Requirements 만들기
- 배포하고 싶은 가상환경으로 activate 하여 아래의 명령어를 실행한다.

```console
source activate yourenv
pip freeze > requirements.txt
```

<br>

### 3. Requirements 설치하기
- 설치하고 싶은 가상환경으로 activate 하여 아래의 명령어를 실행하여 requirements의 패키지를 설치한다.

```console
source activate newserver
pip install -r requirements.txt
```

