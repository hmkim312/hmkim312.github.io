---
title: Beautiful Soup을 이용한 네이버 영화 평점 크롤링
author: HyunMin Kim
date: 2020-10-22 11:10:00 0000
categories: [Python, Crawling]
tags: [Tag, Beautifulsoup, Naver Movie]
---

## 1. 네이버 영화 평점
---
### 1.1 네이버 영화 평점

- <https://movie.naver.com/movie/sdb/rank/rmovie.nhn?sel=cur&date=20201021>{:target="_blank"}
- 네이버 영화에서 영화 평점을 크롤링 해보도록 해보자
- 학습용으로 서버에부하되지않을 정도로만 크롤링하자. 너무많이 크롤링하면 네이버측에서 제제가 들어올수 있다.
- 항상 크롤링은 robots.txt를 확인하자

<br>

### 1.2 URL 보기


```python
'https://movie.naver.com/movie/sdb/rank/rmovie.nhn?sel=cur&date=20201021'
```




    'https://movie.naver.com/movie/sdb/rank/rmovie.nhn?sel=cur&date=20201021'



- URL 맨 뒤에 20201021은 date 형식으로 보임, 해당 날짜를 조금씩 바꾸면 다른 페이지에 접속이 가능
- 이처럼 웹페이지 URL에는 많은 정보가 담겨있음

<br>

### 1.3 한페이지 보기 with Beautiful Soup


```python
from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd

url = 'https://movie.naver.com/movie/sdb/rank/rmovie.nhn?sel=cur&date=20201021'
page = urlopen(url)

soup = BeautifulSoup(page, 'html.parser')
soup
```




    
    <!DOCTYPE html>
    
    <html lang="ko">
    <head>
    <meta content="text/html; charset=utf-8" http-equiv="Content-Type"/>
    <meta content="IE=edge" http-equiv="X-UA-Compatible"/>
    <meta content="http://imgmovie.naver.com/today/naverme/naverme_profile.jpg" property="me2:image">
    <meta content="네이버영화 " property="me2:post_tag">
    <meta content="네이버영화" property="me2:category1"/>
    <meta content="" property="me2:category2"/>
    <meta content="랭킹 : 네이버 영화" property="og:title"/>
    <meta content="영화, 영화인, 예매, 박스오피스 랭킹 정보 제공" property="og:description"/>
    <meta content="article" property="og:type"/>
    <meta content="https://movie.naver.com/movie/sdb/rank/rmovie.nhn?sel=cur&amp;date=20201021" property="og:url"/>
    <meta content="http://static.naver.net/m/movie/icons/OG_270_270.png" property="og:image"/><!-- http://static.naver.net/m/movie/im/navermovie.jpg -->
    <meta content="http://imgmovie.naver.com/today/naverme/naverme_profile.jpg" property="og:article:thumbnailUrl"/>
    <meta content="네이버 영화" property="og:article:author"/>
    <meta content="https://movie.naver.com/" property="og:article:author:url"/>
    <link href="https://ssl.pstatic.net/static/m/movie/icons/naver_movie_favicon.ico" rel="shortcut icon" type="image/x-icon"/>
    <title>랭킹 : 네이버 영화</title>
    <link href="/common/css/movie_tablet.css?20201015140005" rel="stylesheet" type="text/css"/>
    ...
    
    window.addEventListener('pageshow', function(event) { lcs_do(); });
    
    document.addEventListener('click', function (event) {
    	var welSource = event.srcElement;	// jindo.$Element(oEvent.element);
    	if (!document.getElementById("gnb").contains(welSource)) {
    		gnbAllLayerClose();
    	}
    });
    </script>
    <!-- //Footer -->
    </div>
    </body>
    </html> 



- html형식을 beautifulsoup을 사용하여 가져옴

<br>

<img src="https://user-images.githubusercontent.com/60168331/96837405-57203f80-1481-11eb-925f-f340be6dab36.png">

- 위의 내용 중 영화제목과 평점을 가져옴

<br>

### 1.4 영화제목 가져오기

<img src="https://user-images.githubusercontent.com/60168331/96837629-b4b48c00-1481-11eb-9959-eae5d88f4447.png">


- 크롬 개발자 도구로 확인해 본 결과, div 태그 tit5 클래스가 영화 제목


```python
soup.find_all('div', 'tit5')[0].a.string
```
    '소년시절의 너'



- BeautifulSoup의 find_all 명령어로 제목을 모두 찾을수 있음.
- 그중 첫번째 ([0])인 내용만 가져왔고, string 형식인 제목만 가져옴

<br>


```python
movie_name = [i.a.string for i in soup.find_all('div', 'tit5')]
movie_name
```


    ['소년시절의 너',
     '브레이크 더 사일런스: 더 무비',
     '울지마 톤즈',
     '다시 태어나도 우리',
     '언더독',
     '그대, 고맙소 : 김호중 생애 첫 팬미팅 무비',
     '우리들',
     '스파이더맨: 뉴 유니버스',
     '톰보이',
     '사랑과 영혼',
     '파수꾼',
     '제리 맥과이어',
     '삼진그룹 영어토익반',
     '공범자들',
     '타오르는 여인의 초상',
     '박하사탕',
     '인생 후르츠',
     '남매의 여름밤',
     '윤희에게',
     '비투스',
     '아웃포스트',
     '담보',
     '너의 이름은.',
     '소공녀',
     '벌새',
     '마미',
     '브리짓 존스의 일기',
     '찬실이는 복도 많지',
     '69세',
     '라라랜드',
     '기생충',
     '아무르',
     '로렌스 애니웨이 ',
     '3:10 투 유마',
     '검객',
     '주디',
     '리스본행 야간열차',
     '테넷',
     '경계선',
     '프란시스 하',
     '신문기자',
     '위크엔드 인 파리',
     '블레이드 러너 2049',
     '페이트 스테이 나이트 헤븐즈필 제2장 로스트 버터플라이',
     '환상의 빛',
     '날씨의 아이',
     '라붐',
     '21 브릿지: 테러 셧다운',
     '한여름의 판타지아',
     '다만 악에서 구하소서']



- 리스트 컴프리행션을 사용하여 한 페이지의 영화 제목을 모두 가져옴

<br>

### 1.4 평점 가져오기

<img src = 'https://user-images.githubusercontent.com/60168331/96838204-84212200-1482-11eb-95d8-cf40a98d801c.png'>

- td 태그에 point 클래스가 영화 평점

<br>


```python
soup.find_all('td', 'point')[0].string
```




    '9.39'



- 영화 제목 가져왔을떄랑 똑같이 find_all과 string을 사용하여 가져옴

<br>


```python
movie_point = [i.string for i in soup.find_all('td', 'point')]
movie_point
```




    ['9.39',
     '9.36',
     '9.35',
     '9.34',
     '9.30',
     '9.29',
     '9.26',
     '9.20',
     '9.20',
     '9.19',
     '9.18',
     '9.16',
     '9.15',
     '9.10',
     '9.06',
     '9.03',
     '9.02',
     '9.02',
     '8.98',
     '8.94',
     '8.93',
     '8.86',
     '8.78',
     '8.77',
     '8.76',
     '8.69',
     '8.68',
     '8.67',
     '8.63',
     '8.61',
     '8.49',
     '8.48',
     '8.44',
     '8.41',
     '8.36',
     '8.35',
     '8.31',
     '8.27',
     '8.17',
     '8.14',
     '8.10',
     '8.06',
     '7.99',
     '7.98',
     '7.98',
     '7.97',
     '7.93',
     '7.91',
     '7.81',
     '7.64']



- 한 페이지의 평점을 가져옴

<br>

### 1.5 날짜 만들기


```python
'https://movie.naver.com/movie/sdb/rank/rmovie.nhn?sel=cur&date=20201021'
```




    'https://movie.naver.com/movie/sdb/rank/rmovie.nhn?sel=cur&date=20201021'



- 한페이지에서 데이터를 얻게됨
- 아까전에 이야기한 date를 바꾸면 웹 페이지가 바뀌기떄문에, 날짜를 바꿔가며 url을 변경함

<br>


```python
date = pd.date_range('2020.09.01', periods=60, freq='D')
date
```




    DatetimeIndex(['2020-09-01', '2020-09-02', '2020-09-03', '2020-09-04',
                   '2020-09-05', '2020-09-06', '2020-09-07', '2020-09-08',
                   '2020-09-09', '2020-09-10', '2020-09-11', '2020-09-12',
                   '2020-09-13', '2020-09-14', '2020-09-15', '2020-09-16',
                   '2020-09-17', '2020-09-18', '2020-09-19', '2020-09-20',
                   '2020-09-21', '2020-09-22', '2020-09-23', '2020-09-24',
                   '2020-09-25', '2020-09-26', '2020-09-27', '2020-09-28',
                   '2020-09-29', '2020-09-30', '2020-10-01', '2020-10-02',
                   '2020-10-03', '2020-10-04', '2020-10-05', '2020-10-06',
                   '2020-10-07', '2020-10-08', '2020-10-09', '2020-10-10',
                   '2020-10-11', '2020-10-12', '2020-10-13', '2020-10-14',
                   '2020-10-15', '2020-10-16', '2020-10-17', '2020-10-18',
                   '2020-10-19', '2020-10-20', '2020-10-21', '2020-10-22',
                   '2020-10-23', '2020-10-24', '2020-10-25', '2020-10-26',
                   '2020-10-27', '2020-10-28', '2020-10-29', '2020-10-30'],
                  dtype='datetime64[ns]', freq='D')



- Pandas의 date_range를 사용하여 날짜를 생성
- 시작날짜를 적어주고, 만들고 싶은 날짜 갯수와 날짜 형태를 적으면 됨

<br>


```python
print(date[0].strftime('%y-%m-%d'))
print(date[0].strftime('%y.%m.%d'))
print(date[0].strftime('%y%m%d'))
print(date[0].strftime('%Y%m%d'))
```

    20-09-01
    20.09.01
    200901
    20200901


- 날짜형 데이터는 strftime 명령으로 원하는 형태의 문자열로 만들수 있음
- URL에서 필요한 형식은 맨 아래 20200901 형식임

<br>

### 1.6 여러날짜에서 영화제목과 평점가져오기


```python
import time

movie_date = []
movie_name = []
movie_point = []
date = pd.date_range('2020.09.01', periods=45, freq='D')

for today in date:
    html = 'https://movie.naver.com/movie/sdb/rank/rmovie.nhn?sel=cur&date={date}'
    response = urlopen(html.format(date = today.strftime('%Y%m%d')))
    soup = BeautifulSoup(response, 'html.parser')
    
    movie_date.extend([today] * len(soup.find_all('td', 'point')))
    movie_name.extend([i.a.string for i in soup.find_all('div', 'tit5')])
    movie_point.extend([i.string for i in soup.find_all('td', 'point')])
    
    print(str(today))
    time.sleep(0.5)
```

    2020-09-01 00:00:00
    2020-09-02 00:00:00
    2020-09-03 00:00:00
    2020-09-04 00:00:00
    2020-09-05 00:00:00
    2020-09-06 00:00:00
    2020-09-07 00:00:00
    2020-09-08 00:00:00
    2020-09-09 00:00:00
    2020-09-10 00:00:00
    2020-09-11 00:00:00
    2020-09-12 00:00:00
    2020-09-13 00:00:00
    2020-09-14 00:00:00
    2020-09-15 00:00:00
    2020-09-16 00:00:00
    2020-09-17 00:00:00
    2020-09-18 00:00:00
    2020-09-19 00:00:00
    2020-09-20 00:00:00
    2020-09-21 00:00:00
    2020-09-22 00:00:00
    2020-09-23 00:00:00
    2020-09-24 00:00:00
    2020-09-25 00:00:00
    2020-09-26 00:00:00
    2020-09-27 00:00:00
    2020-09-28 00:00:00
    2020-09-29 00:00:00
    2020-09-30 00:00:00
    2020-10-01 00:00:00
    2020-10-02 00:00:00
    2020-10-03 00:00:00
    2020-10-04 00:00:00
    2020-10-05 00:00:00
    2020-10-06 00:00:00
    2020-10-07 00:00:00
    2020-10-08 00:00:00
    2020-10-09 00:00:00
    2020-10-10 00:00:00
    2020-10-11 00:00:00
    2020-10-12 00:00:00
    2020-10-13 00:00:00
    2020-10-14 00:00:00
    2020-10-15 00:00:00


- 일단 20년 9월 1일 ~ 10월 15일까지의 영화 평점을 가져옴
- 한페이지를 크롤링하고 또 바로 크롤링하면 과부화 및 접속 차단등이 걸릴수 있으니 time.sleep을 걸어주지

<br>


```python
len(movie_date), len(movie_name), len(movie_point)
```




    (2158, 2158, 2158)



- 총 2158개의 영화 평점을 가져옴

<br>

### 1.7 데이터 프레임 생성


```python
movie = pd.DataFrame({'date' : movie_date, 'name': movie_name, 'point' : movie_point})
movie.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>name</th>
      <th>point</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2153</th>
      <td>2020-10-15</td>
      <td>언힌지드</td>
      <td>6.69</td>
    </tr>
    <tr>
      <th>2154</th>
      <td>2020-10-15</td>
      <td>죽지않는 인간들의 밤</td>
      <td>6.62</td>
    </tr>
    <tr>
      <th>2155</th>
      <td>2020-10-15</td>
      <td>강철비2: 정상회담</td>
      <td>5.01</td>
    </tr>
    <tr>
      <th>2156</th>
      <td>2020-10-15</td>
      <td>국제수사</td>
      <td>4.87</td>
    </tr>
    <tr>
      <th>2157</th>
      <td>2020-10-15</td>
      <td>뮬란</td>
      <td>4.20</td>
    </tr>
  </tbody>
</table>
</div>



<br>


```python
movie.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2158 entries, 0 to 2157
    Data columns (total 3 columns):
     #   Column  Non-Null Count  Dtype         
    ---  ------  --------------  -----         
     0   date    2158 non-null   datetime64[ns]
     1   name    2158 non-null   object        
     2   point   2158 non-null   object        
    dtypes: datetime64[ns](1), object(2)
    memory usage: 50.7+ KB


- 평점은 float으로 변경

<br>


```python
movie['point'] = movie['point'].astype(float)
movie.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2158 entries, 0 to 2157
    Data columns (total 3 columns):
     #   Column  Non-Null Count  Dtype         
    ---  ------  --------------  -----         
     0   date    2158 non-null   datetime64[ns]
     1   name    2158 non-null   object        
     2   point   2158 non-null   float64       
    dtypes: datetime64[ns](1), float64(1), object(1)
    memory usage: 50.7+ KB


- 평점타입 변경 완료

<br>


```python
movie.to_csv('./data/naver_movie_points_20201022.csv', sep = ',', encoding = 'utf-8')
```

- 크롤링된 데이터는 csv파일로 저장함
- 만일 커널이 재시작되면 크롤링한 데이터는 날라감
- 해당 파일은 github에 올려둠 <https://raw.githubusercontent.com/hmkim312/datas/main/navermoviepoints/naver_movie_points_20201022.csv>{target:'_blank'}

<br>

## 2. 데이터 Preprocessing
---
### 2.1 Data Load


```python
import numpy as np
import pandas as pd

movie = pd.read_csv('https://raw.githubusercontent.com/hmkim312/datas/main/navermoviepoints/naver_movie_points_20201022.csv', index_col = 0)
movie.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>name</th>
      <th>point</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2153</th>
      <td>2020-10-15</td>
      <td>언힌지드</td>
      <td>6.69</td>
    </tr>
    <tr>
      <th>2154</th>
      <td>2020-10-15</td>
      <td>죽지않는 인간들의 밤</td>
      <td>6.62</td>
    </tr>
    <tr>
      <th>2155</th>
      <td>2020-10-15</td>
      <td>강철비2: 정상회담</td>
      <td>5.01</td>
    </tr>
    <tr>
      <th>2156</th>
      <td>2020-10-15</td>
      <td>국제수사</td>
      <td>4.87</td>
    </tr>
    <tr>
      <th>2157</th>
      <td>2020-10-15</td>
      <td>뮬란</td>
      <td>4.20</td>
    </tr>
  </tbody>
</table>
</div>



- index_col = 0 옵션을 넣어서 인덱스를 불러오지 않도록 함

<br>

### 2.2 평점 합산


```python
movie_unique = pd.pivot_table(movie, index=['name'], aggfunc=np.sum)
movie_unique.sort_values('point', ascending = False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>point</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>소년시절의 너</th>
      <td>422.21</td>
    </tr>
    <tr>
      <th>사랑과 영혼</th>
      <td>413.48</td>
    </tr>
    <tr>
      <th>제리 맥과이어</th>
      <td>412.20</td>
    </tr>
    <tr>
      <th>극장판 짱구는 못말려: 신혼여행 허리케인~ 사라진 아빠!</th>
      <td>393.58</td>
    </tr>
    <tr>
      <th>브리짓 존스의 일기</th>
      <td>390.57</td>
    </tr>
    <tr>
      <th>69세</th>
      <td>388.31</td>
    </tr>
    <tr>
      <th>500일의 썸머</th>
      <td>378.90</td>
    </tr>
    <tr>
      <th>라라랜드</th>
      <td>378.84</td>
    </tr>
    <tr>
      <th>테넷</th>
      <td>372.53</td>
    </tr>
    <tr>
      <th>타오르는 여인의 초상</th>
      <td>371.08</td>
    </tr>
  </tbody>
</table>
</div>



- 영화 이름으로 인덱스를 잡고, 점수를 합산 후 내림차순 10개를 출력함, best 10

<br>

### 2.3 DataFrame Query


```python
movie.query('name == ["테넷"]')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>name</th>
      <th>point</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32</th>
      <td>2020-09-01</td>
      <td>테넷</td>
      <td>8.37</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2020-09-02</td>
      <td>테넷</td>
      <td>8.36</td>
    </tr>
    <tr>
      <th>129</th>
      <td>2020-09-03</td>
      <td>테넷</td>
      <td>8.34</td>
    </tr>
    <tr>
      <th>179</th>
      <td>2020-09-04</td>
      <td>테넷</td>
      <td>8.33</td>
    </tr>
    <tr>
      <th>225</th>
      <td>2020-09-05</td>
      <td>테넷</td>
      <td>8.34</td>
    </tr>
    <tr>
      <th>277</th>
      <td>2020-09-06</td>
      <td>테넷</td>
      <td>8.31</td>
    </tr>
    <tr>
      <th>326</th>
      <td>2020-09-07</td>
      <td>테넷</td>
      <td>8.29</td>
    </tr>
    <tr>
      <th>373</th>
      <td>2020-09-08</td>
      <td>테넷</td>
      <td>8.28</td>
    </tr>
    <tr>
      <th>418</th>
      <td>2020-09-09</td>
      <td>테넷</td>
      <td>8.28</td>
    </tr>
    <tr>
      <th>463</th>
      <td>2020-09-10</td>
      <td>테넷</td>
      <td>8.28</td>
    </tr>
    <tr>
      <th>509</th>
      <td>2020-09-11</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>560</th>
      <td>2020-09-12</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>610</th>
      <td>2020-09-13</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>656</th>
      <td>2020-09-14</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>706</th>
      <td>2020-09-15</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>750</th>
      <td>2020-09-16</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>794</th>
      <td>2020-09-17</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>841</th>
      <td>2020-09-18</td>
      <td>테넷</td>
      <td>8.28</td>
    </tr>
    <tr>
      <th>887</th>
      <td>2020-09-19</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>932</th>
      <td>2020-09-20</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>975</th>
      <td>2020-09-21</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>1018</th>
      <td>2020-09-22</td>
      <td>테넷</td>
      <td>8.26</td>
    </tr>
    <tr>
      <th>1055</th>
      <td>2020-09-23</td>
      <td>테넷</td>
      <td>8.26</td>
    </tr>
    <tr>
      <th>1096</th>
      <td>2020-09-24</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>1146</th>
      <td>2020-09-25</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>1196</th>
      <td>2020-09-26</td>
      <td>테넷</td>
      <td>8.26</td>
    </tr>
    <tr>
      <th>1247</th>
      <td>2020-09-27</td>
      <td>테넷</td>
      <td>8.26</td>
    </tr>
    <tr>
      <th>1293</th>
      <td>2020-09-28</td>
      <td>테넷</td>
      <td>8.26</td>
    </tr>
    <tr>
      <th>1340</th>
      <td>2020-09-29</td>
      <td>테넷</td>
      <td>8.26</td>
    </tr>
    <tr>
      <th>1392</th>
      <td>2020-09-30</td>
      <td>테넷</td>
      <td>8.26</td>
    </tr>
    <tr>
      <th>1442</th>
      <td>2020-10-01</td>
      <td>테넷</td>
      <td>8.26</td>
    </tr>
    <tr>
      <th>1492</th>
      <td>2020-10-02</td>
      <td>테넷</td>
      <td>8.26</td>
    </tr>
    <tr>
      <th>1542</th>
      <td>2020-10-03</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>1589</th>
      <td>2020-10-04</td>
      <td>테넷</td>
      <td>8.26</td>
    </tr>
    <tr>
      <th>1643</th>
      <td>2020-10-05</td>
      <td>테넷</td>
      <td>8.26</td>
    </tr>
    <tr>
      <th>1698</th>
      <td>2020-10-06</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>1747</th>
      <td>2020-10-07</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>1800</th>
      <td>2020-10-08</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>1849</th>
      <td>2020-10-09</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>1899</th>
      <td>2020-10-10</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>1945</th>
      <td>2020-10-11</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>1990</th>
      <td>2020-10-12</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>2038</th>
      <td>2020-10-13</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>2088</th>
      <td>2020-10-14</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>2138</th>
      <td>2020-10-15</td>
      <td>테넷</td>
      <td>8.27</td>
    </tr>
  </tbody>
</table>
</div>



- DataFrame Query로 검색을 해볼수 있음

<br>

### 2.4 날짜별 영화 평점 변화 그리기


```python
import matplotlib.pyplot as plt

plt.figure(figsize=(24, 8))
plt.plot(movie.query('name == ["테넷"]')['date'],
         movie.query('name == ["테넷"]')['point'])
plt.legend(labels = ['point'])
plt.xticks(rotation=45)
plt.grid()
plt.show()
```

<img src = 'https://user-images.githubusercontent.com/60168331/96848498-7920be80-148f-11eb-8655-c62fb95e02d4.png'>


- 영화 테넷의 평점 변화를 봄
- x 축의 길이가 너무 길어서 rotation을 45로 해줌
- 그래프의 곡선이 많이 움직이는것 같지만, 사실 y축을 보면 그렇게 크지 않음 최고점과 최저점이 0.1점 차이

<br>

### 2.5 영화 정리


```python
movie_pivot = movie.pivot_table(movie, index = ['date'], columns = ['name'])
movie_pivot.columns = movie_pivot.columns.droplevel([0])
movie_pivot.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>name</th>
      <th>500일의 썸머</th>
      <th>69세</th>
      <th>가버나움</th>
      <th>감쪽같은 그녀</th>
      <th>강철비2: 정상회담</th>
      <th>검객</th>
      <th>경계선</th>
      <th>국제수사</th>
      <th>그대, 고맙소 : 김호중 생애 첫 팬미팅 무비</th>
      <th>그래비티</th>
      <th>...</th>
      <th>포드 V 페라리</th>
      <th>폭스캐처</th>
      <th>프란시스 하</th>
      <th>피아노</th>
      <th>피아니스트</th>
      <th>피아니스트의 전설</th>
      <th>피터와 드래곤</th>
      <th>하녀</th>
      <th>항거:유관순 이야기</th>
      <th>홀리 모터스</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-10-11</th>
      <td>8.42</td>
      <td>8.63</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.36</td>
      <td>8.17</td>
      <td>NaN</td>
      <td>9.47</td>
      <td>8.29</td>
      <td>...</td>
      <td>9.49</td>
      <td>NaN</td>
      <td>8.14</td>
      <td>NaN</td>
      <td>9.32</td>
      <td>NaN</td>
      <td>8.25</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-10-12</th>
      <td>8.42</td>
      <td>8.63</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.36</td>
      <td>8.17</td>
      <td>NaN</td>
      <td>9.45</td>
      <td>NaN</td>
      <td>...</td>
      <td>9.49</td>
      <td>NaN</td>
      <td>8.14</td>
      <td>NaN</td>
      <td>9.32</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.52</td>
    </tr>
    <tr>
      <th>2020-10-13</th>
      <td>8.42</td>
      <td>8.63</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.01</td>
      <td>8.35</td>
      <td>8.17</td>
      <td>4.89</td>
      <td>9.46</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.14</td>
      <td>NaN</td>
      <td>9.32</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.52</td>
    </tr>
    <tr>
      <th>2020-10-14</th>
      <td>8.42</td>
      <td>8.63</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.01</td>
      <td>8.36</td>
      <td>8.17</td>
      <td>4.88</td>
      <td>9.44</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.14</td>
      <td>NaN</td>
      <td>9.32</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.52</td>
    </tr>
    <tr>
      <th>2020-10-15</th>
      <td>8.42</td>
      <td>8.63</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.01</td>
      <td>8.36</td>
      <td>8.17</td>
      <td>4.87</td>
      <td>9.43</td>
      <td>NaN</td>
      <td>...</td>
      <td>9.49</td>
      <td>NaN</td>
      <td>8.14</td>
      <td>NaN</td>
      <td>9.32</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.52</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 132 columns</p>
</div>



- 크롤링한 영화를 컬럼으로 하게 pivot함

<br>

### 2.6 보고싶은 영화들의 평점들 시각화


```python
movie_pivot.columns
```




    Index(['500일의 썸머', '69세', '가버나움', '감쪽같은 그녀', '강철비2: 정상회담', '검객', '경계선', '국제수사',
           '그대, 고맙소 : 김호중 생애 첫 팬미팅 무비', '그래비티',
           ...
           '포드 V 페라리', '폭스캐처', '프란시스 하', '피아노', '피아니스트', '피아니스트의 전설', '피터와 드래곤',
           '하녀', '항거:유관순 이야기', '홀리 모터스'],
          dtype='object', name='name', length=132)




```python
targer_col = ['극장판 짱구는 못말려: 신혼여행 허리케인~ 사라진 아빠!', '테넷', '라라랜드', '동주']

plt.figure(figsize=(12,8))
plt.plot(movie_pivot[targer_col])
plt.legend(targer_col, loc = 'best')
plt.tick_params(bottom = False, labelbottom = False)
plt.show()
```


<img src= 'https://user-images.githubusercontent.com/60168331/96848695-b2f1c500-148f-11eb-8621-94367b889eae.png'>


- 보고싶은 영화의 평점만 골라서 비교해볼수 있음
- 중간에 선이 끊긴것은 평점 데이터가 없는것

<br>

### 2.7 엑셀로 저장


```python
movie_pivot.to_excel('./data/naver_movie_points_pivot_20201022.xlsx')
```

<img src="https://user-images.githubusercontent.com/60168331/96846739-41b11280-148d-11eb-9e9e-3a0768547792.png">

- to_excel을 사용하여 저장
- 이런식으로 저장됨
