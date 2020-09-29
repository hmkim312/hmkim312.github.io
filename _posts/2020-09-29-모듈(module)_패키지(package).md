---
title: 모듈(module)과 패키지(package)
author: HyunMin Kim
date: 2020-09-29 11:30:00 0000
categories: [Python, Basic]
tags: [Module, Package]
---

## 1. 모듈과 패키지
---
### 1.1 모듈과 패키지란
- 모듈 : 변수, 함수, 클래스를 모아놓은(.py) 확장자를 가진 파일 클래스보다 조금 더 큰 규모
- 패키지 : 모듈의 기능을 디렉토리별로 정리해 놓은 개념
- 아래의 내용들은 jupyter 환경에서 실습된 내용임 

<br>

## 2. 모듈
---
### 2.1 모듈 생성

```python
%%writefile dss2.py

num = 1234

def disp1(msg):
    print('disp1', msg)
    
def disp2(msg):
    print('disp2', msg)
    
class Calc:
    def plus(self, *args):
        return sum(args)
```
    Writing dss2.py

- `%%writefile` (매직 메소드)를 사용하여 해당 셀을 py file로 만든다.
- num이라는 변수, dips1,2 함수, Calc라는 클래스를 포함한 dss2.py라는 모듈을 생성함

<br>

```python
!ls
```
    dss2.py

- !ls를 사용하면 해당 디렉토리에 있는 파일을 보여주게 된다.
- !ls 를 사용하여 방금 생성한 dss2.py를 보았음
- 터미널에서 사용하는 명령어와 같다.

<br>

### 2.2 모듈 호출

```python
import dss2
```

```python
whos
```

    Variable   Type      Data/Info
    ------------------------------
    dss2       module    <module 'dss2' from '/Use<...>01_Python_basic/dss2.py'>

- import 모듈명
- 방금 생성한 dss2.py 모듈을 import로 불러옴
- whos 명령어로 현재 불러온 dss2 모듈을 확인할수 있음

<br>


#### 2.2.1 모듈 안에 특정 함수, 변수, 클래스를 호출

```python
from dss2 import num, disp2
```


- from 모듈명 import 특정 함수, 변수 클래스 명
- form dss2 import num, disp2 로 num과 disp2를 호출함

<br>


#### 2.2.2 모듈의 모든 변수를 가져옴

```python

from dss2 import *
```

- from 모듈명  import * 로 모든 변수를 가져옴

<br>

### 2.3 모듈 사용

```python
dss2.num
```
    1234

- num 변수 사용
- 변수는 모듈명.변수명으로 사용

<br>


```python
dss2.disp1('python')
```
    disp1 python

- disp1 메서드(함수) 사용
- 메서드는 모듈명.메서드(파라미터)로 사용
<br>

```python
calc = dss2.Calc()
```

```python
calc.plus(1,2,3,4)
```
    10

- Calc 클래스 사용
- 클래스는 모듈명.클래스명으로 객체를 생성 후 클래스의 메서드를 사용함

<br>

## 3. Package
--- 
### 3.1 패키지 생성
#### 3.1.1 디렉토리 생성

```python
!mkdir -p school2/dss
!mkdir -p school2/web
```

- !mkdir 을 사용하여 디렉토리를 생성함 
- !mkdir (windows)
- !mkdir -p (mac) 
- !mkdir은 -p 옵션을 넣어야 디렉토리가 생성됨

<br>

#### 3.1.2 `__init__.py`파일 추가

```python

!touch school2/dss/__init__.py
!touch school2/web/__init__.py
```
- 패키지 사용시 디렉토리에 __init__.py 파일을 추가

<br>

#### 3.1.3 디렉토리 구조 확인

```python
!tree school2
```

    [01;34mschool2[00m
    ├── [01;34mdss[00m
    │   └── __init__.py
    └── [01;34mweb[00m
        └── __init__.py
    
    2 directories, 2 files

- !tree로 해당 디렉토리의 구조를 확인 가능
- !tree (mac)
- !tree -f (windows)
- tree는 설치해야함 (homebrew, pip)

<br>

#### 3.1.4 디렉토리에 모듈 생성

```python
%%writefile school2/dss/data1.py

def plus(*args):
    print("data1")
    return sum(args)
```
    Writing school2/dss/data1.py

<br>

```python
%%writefile school2/dss/data2.py

def plus2(*args):
    print("data2")
    return sum(args)
```
    Writing school2/dss/data2.py

<br>

```python
%%writefile school2/web/url.py

def make(url):
    return url if url[:7] == "http://" else "http://" + url
```
    Writing school2/web/url.py

- school2 디렉토리 아래에 있는 dss와 web에 넣을 모듈들을 생성함
- data1, data2는 주어진 파라미터를 모두 더하는 모듈
- make 모듈은 url을 받으면 http:// 가 없으면 http:// 를 붙여주는 모듈임

<br>

### 3.2 패키지 호출

- 모든 패키지 및 모듈은 불러오는 파일(ipynb, py)과 같은 위치에 있어야함

```python
import school2.dss.data1
```

- school2 디렉토리 아래에있는 dss의 디렉토리의 data1을 모듈을 호출함

<br>

```python
import school2.dss.data1 as dss
```

- as는 alias의 약자로 해당 모듈을 불러오는데 별칭으로 명하는것
- 앞으로 해당 모듈을 부를때 dss로 불러서 사용가능함

<br>

```python
from school.web import url
```

- `school.web` : 디렉토리
- url : 모듈
- import 뒤에는 모듈이 와야함

<br>

#### 3.2.1 어디서나 import 하기

```python
import sys

for path in sys.path:
    print(path)
```
    /opt/anaconda3/lib/python38.zip
    /opt/anaconda3/lib/python3.8
    /opt/anaconda3/lib/python3.8/lib-dynload
    
    /opt/anaconda3/lib/python3.8/site-packages
    /opt/anaconda3/lib/python3.8/site-packages/aeosa
    /opt/anaconda3/lib/python3.8/site-packages/IPython/extensions
    /Users/hmkim/.ipython

- 특정 디렉토리에 있는 패키지는 어디에서나 import 가능하며, 설치된 경로 등에 따라서 모두 다름
- sys 패키지 및 for문을 이용하여 path를 출력함
- 출력되는 라이브러리 저장소 중에 사용가능한 패키지가 설치된 폴더를 찾음 

<br>

```python
!ls /opt/anaconda3/lib/python3.8/
```
    LICENSE.txt
    __future__.py
    __phello__.foo.py
    [1m[36m__pycache__[m[m
    _bootlocale.py
    _collections_abc.py
    _compat_pickle.py
    _compression.py
    _dummy_thread.py
    _markupbase.py
    _osx_support.py
    _py_abc.py
    xdrlib.py
    [1m[36mxml[m[m
    [1m[36mxmlrpc[m[m
    zipapp.py
    zipfile.py
    zipimport.py
    ...

- /opt/anaconda3/lib/python3.8 가 패키지 있는 디렉토리

<br>

```python
packages = !ls /opt/anaconda3/lib/python3.8/
```

```python
len(packages)
```
    212

- 쉘커맨드도 변수로 설정가능
- 위의 디렉토리에 저장된 패키지의 갯수는 212개 

<br>

### 3.3 `setup.py` 패키지 설치 파일 만들기
- setup.py를 작서해서 패키지를 설치해서 사용해봄
- setup tools

```python
!tree school2/
```

    [01;34mschool2/[00m
    ├── [01;34mdss[00m
    │   ├── __init__.py
    │   ├── [01;34m__pycache__[00m
    │   │   ├── __init__.cpython-38.pyc
    │   │   └── data1.cpython-38.pyc
    │   ├── data1.py
    │   └── data2.py
    └── [01;34mweb[00m
        ├── __init__.py
        └── url.py
    
    3 directories, 7 files

- !tree로 패키지를 만들 디렉토리의 구조 확인

<br>

```python
%%writefile school2/setup.py

from setuptools import setup, find_packages

setup(
    name="dss",
    packages = find_packages(),
    include_package_data = True,
    version = "0.0.1",
    author = "HyunMin Kim",
    author_email = "이메일@도메인",
    zip_safe=False,
)
```
    Overwriting school2/setup.py

- setuptools를 이용하여 설치 파일을 생성
- name, version, author, email은 본인에 맞게 작성 하면됨

<br>

```python
!tree school2/
```

    [01;34mschool2/[00m
    ├── [01;34mdss[00m
    │   ├── __init__.py
    │   ├── [01;34m__pycache__[00m
    │   │   ├── __init__.cpython-38.pyc
    │   │   └── data1.cpython-38.pyc
    │   ├── data1.py
    │   └── data2.py
    ├── [01;34mdss.egg-info[00m
    │   ├── PKG-INFO
    │   ├── SOURCES.txt
    │   ├── dependency_links.txt
    │   ├── not-zip-safe
    │   └── top_level.txt
    ├── setup.py
    └── [01;34mweb[00m
        ├── __init__.py
        └── url.py
    
    4 directories, 13 files

- `setup.py`가 생성된것을 확인할수 있다. 

<br>

### 3.4 패키지 설치
- 패키지 설치, setup.py 있는 폴더에서 터미널 실행 후
- ``` school2 $ python setup.py develop ``` 입력
- 커널 리스타트
- develop : 개발자모드, 코드를 수정하면 설치된 패키지도 같이 수정
- build : 일반모드, 코드를 수정하면 다시 설치해야 수정된 코드가 적용

<br>

#### 3.4.1 패키지 설치 확인

```python

!pip list | grep dss
```
    dss                                0.0.1                 설치된 경로

- !pip list | grep dss 
- grep dss는 dss 이름을 찾음
- 자주사용하는 함수나 클래스는 패키지로 생성하여 설치 후 사용하는것을 권장함
