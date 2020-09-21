---
title: Git Contribution(잔디) 안올라갈때
author: HyunMin Kim
date: 2020-09-21 12:00:00 0000
categories: [Blog, Github]
tags: [Blog, Github, Contribution]
---


## 1. Git Contribution(잔디) 안올라갈때
### 1.1 git config에 user.name, user.email이 안맞을경우
-  **잔디가 안올라가는 폴더**에서 name과 email이 등록되어 있는지 확인
```
cd /잔디가 안올라가는 폴더로 이동
git config --list
```

- 전역으로 user.name, user.eamil 등록
```
git config --global user.name 내유저이름
git config --global user.email 내 이메일
```

- gitconfig에 user.name과 user.email이 등록되어있는지 확인
```
vi ~/.gitconfig
```

- 아래처럼 등록안되어있으면 작성(vi 에디터 사용)
```
[user]
    name = 내유저이름
    email = 내이메일
```

- 이후 commit 후 push하여 확인하면 대부분 해결됨

### 1.2 Fork한 레파지토리일 경우
- Fork한 레파지토리는 아무리 commit해도 잔디가 심어지지 않음
- Github에 Fork한 레파지토리 삭제
- 똑같은 이름으로 레파지토리 생성
- 로컬(내 컴퓨터에 실제 파일이 있는 저장소)에서 init, add, commit, push
``` 
git init
git add .
git commit -m 'fixed : my repository'
git push
```
- 요약하자면 Github 사이트에서 Fork한 레파지토리는 삭제하고, 신규 레파지토리를 생성 후에 다시 init하여 push하는것
- 다시보면 잔디가 잘 심어지는것을 알수 있음 .. 휴 다행
