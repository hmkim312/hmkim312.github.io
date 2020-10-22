---
title: Beautiful Soup을 이용한 시카고 샌드위치 맛집 정보 추출
author: HyunMin Kim
date: 2020-10-22 11:10:00 0000
categories: [Python, Crawling]
tags: [Tag, Beautifulsoup, chicago Magazine]
---

## 1. 시카고 매거진
---
### 1.1 시카고 매거진

<img src="https://user-images.githubusercontent.com/60168331/96872753-fe679b80-14ae-11eb-947d-1bca17de95d0.png">

<https://www.chicagomag.com/Chicago-Magazine/November-2012/Best-Sandwiches-Chicago/>{:target="_blank"}
- 미국 시카고 매거진의 베스트 50개의 샌드위치 맛집 리스트
- 메뉴와 가게 이름이 정리

<br>

## 1.2 가게 상세 페이지

<img src="https://user-images.githubusercontent.com/60168331/96872944-3b339280-14af-11eb-8f28-eacfa460ede8.png">

- 각각소개한 50개의 페이지에 들어가면 가게 주소와 대표메뉴의 가격
- 즉, 총 51개의 페이지에서, 가게이름, 대표메뉴, 대표메뉴의 가격, 가게주소를 수집

<br>

## 2. 기본 페이지
---
### 2.1 기본 페이지 파싱


```python
from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd

url_base = 'https://www.chicagomag.com'
url_sub = '/Chicago-Magazine/November-2012/Best-Sandwiches-Chicago/'
url = url_base + url_sub

html = urlopen(url)
soup = BeautifulSoup(html, 'html.parser')

soup
```




    
    <!DOCTYPE doctype html>
    
    <html lang="en">
    <head>
    <!-- Urbis magnitudo. Fabulas magnitudo. -->
    <meta charset="utf-8"/>
    <style>a.edit_from_site {display: none !important;}</style>
    <title>
      The 50 Best Sandwiches in Chicago |
      Chicago magazine
          |  November 2012
        </title>
    ...
    </script>
    <!--[if lt IE 9]>
    <script  type="text/javascript" language="JavaScript" src="/core/media/themes/Respond/js/respond.js?ver=1473876729"></script>
    <![endif]-->
    <script language="JavaScript" src="/core/media/js/base.js?ver=1473876728" type="text/javascript"></script>
    <script language="JavaScript" src="/core/media/themes/Respond/js/bootstrap.min.js?ver=1473876729" type="text/javascript"></script>
    <script language="JavaScript" src="//maps.googleapis.com/maps/api/js?v=3.exp&amp;sensor=false" type="text/javascript"></script>
    <script language="JavaScript" src="/theme_overrides/Respond/js/interstitial.js?ver=1524154906" type="text/javascript"></script>
    <script language="JavaScript" src="/theme_overrides/Respond/js/newsletter-subscribe.js?ver=1524850607" type="text/javascript"></script>
    <script language="JavaScript" src="/theme_overrides/Respond/js/RivistaGoogleDFP.js?ver=1447178886" type="text/javascript"></script>
    <!-- godengo-monitor --></body>
    </html>



- 기본 시카고 매거진의 url과 2012년 11월 시카고 매거진의 50개 샌드위치 url을 합쳐서 완성된 url을 만듬
- 이후 Beautifulsoup을 사용하여 html로 파싱함

<br>

### 2.2 50개 가게 정보 가져오기

<img src="https://user-images.githubusercontent.com/60168331/96873521-04aa4780-14b0-11eb-8e2b-3b413612cb32.png">


```python
len(soup.find_all('div', 'sammy'))
```




    50



- div의 sammy 클래스를 이용해 총 50개의 가게 정보를 가져옴

<br>

### 2.3 가게 정보


```python
print(soup.find_all('div', 'sammy')[0])
```

    <div class="sammy" style="position: relative;">
    <div class="sammyRank">1</div>
    <div class="sammyListing"><a href="/Chicago-Magazine/November-2012/Best-Sandwiches-in-Chicago-Old-Oak-Tap-BLT/"><b>BLT</b><br>
    Old Oak Tap<br>
    <em>Read more</em> </br></br></a></div>
    </div>


- 각 가게 정보에 랭킹, 가게이름, 메뉴, 상세페이지로연결되는 url 이 포함 되어있음

<br>

### 2.4 랭킹 가져오기


```python
tmp_one = soup.find_all('div', 'sammy')[0]
tmp_one.find(class_ = 'sammyRank').get_text()
```
    '1'



- find 메소드와 sammyRank 클래스를 이용하여 랭킹을 확보

<br>
    
### 2.5 가게이름과 메뉴 가져오기


```python
tmp_one.find(class_ = 'sammyListing').get_text()
```




    'BLT\r\nOld Oak Tap\nRead more '



- 가게이름(Old Oak Tap)과 메뉴이름(BLT)은 sammyListing 클래스에 한번에 있음

<br>


```python
import re

tmp_string = tmp_one.find(class_ = 'sammyListing').get_text()
print(re.split(('\n|\r\n'), tmp_string)[0])
print(re.split(('\n|\r\n'), tmp_string)[1])
print(re.split(('\n|\r\n'), tmp_string))
```

    BLT
    Old Oak Tap
    ['BLT', 'Old Oak Tap', 'Read more ']


- 가게이름과 메뉴이름은 re 모듈의 split으로 간단히 구분 가능

<br>

### 2.6 상세페이지


```python
tmp_one.find('a')['href']
```




    '/Chicago-Magazine/November-2012/Best-Sandwiches-in-Chicago-Old-Oak-Tap-BLT/'



- 가게의 상세페이지로 가는 url은 상대 경로로, href로 찾으면 됨

<br>


```python
print(url_base)
print(tmp_one.find('a')['href'])
```

    https://www.chicagomag.com
    /Chicago-Magazine/November-2012/Best-Sandwiches-in-Chicago-Old-Oak-Tap-BLT/


- 처음에 설정한 url_base를 사용하여 가게 상세 url과 합치면 될듯 하다.

<br>


```python
from urllib.parse import urljoin
urljoin(url_base, tmp_one.find('a')['href'])
```




    'https://www.chicagomag.com/Chicago-Magazine/November-2012/Best-Sandwiches-in-Chicago-Old-Oak-Tap-BLT/'



- urljoin을 사용하여 url_base와 가게 상제 페이지 주소를 합쳐줌(절대 주소)

<br>


```python
tmp_one = soup.find_all('div', 'sammy')[10].find('a')['href']
print(tmp_one)
urljoin(url_base, tmp_one)
```

    http://www.chicagomag.com/Chicago-Magazine/November-2012/Best-Sandwiches-in-Chicago-Lula-Cafe-Ham-and-Raclette-Panino/


    'http://www.chicagomag.com/Chicago-Magazine/November-2012/Best-Sandwiches-in-Chicago-Lula-Cafe-Ham-and-Raclette-Panino/'



- 11번쨰의 주소도 잘 합쳐짐을 확인함

<br>

### 2.7 50개 메뉴에 적용하기


```python
rank = []
main_menu = []
cafe_name = []
url_add = []

for item in soup.find_all('div', 'sammy'):
    rank.append(item.find(class_ = 'sammyRank').get_text())
    tmp_string = item.find(class_ = 'sammyListing').get_text()
    main_menu.append(re.split(('\n|\r\n'), tmp_string)[0])
    cafe_name.append(re.split(('\n|\r\n'), tmp_string)[1])
    url_add.append(urljoin(url_base, item.find('a')['href']))
```


```python
print(len(rank), len(main_menu), len(cafe_name), len(url_add))
print(main_menu[:5])
print(cafe_name[:5])
url_add[:5]
```

    50 50 50 50
    ['BLT', 'Fried Bologna', 'Woodland Mushroom', 'Roast Beef', 'PB&L']
    ['Old Oak Tap', 'Au Cheval', 'Xoco', 'Al’s Deli', 'Publican Quality Meats']
    ['https://www.chicagomag.com/Chicago-Magazine/November-2012/Best-Sandwiches-in-Chicago-Old-Oak-Tap-BLT/',
     'https://www.chicagomag.com/Chicago-Magazine/November-2012/Best-Sandwiches-in-Chicago-Au-Cheval-Fried-Bologna/',
     'https://www.chicagomag.com/Chicago-Magazine/November-2012/Best-Sandwiches-in-Chicago-Xoco-Woodland-Mushroom/',
     'https://www.chicagomag.com/Chicago-Magazine/November-2012/Best-Sandwiches-in-Chicago-Als-Deli-Roast-Beef/',
     'https://www.chicagomag.com/Chicago-Magazine/November-2012/Best-Sandwiches-in-Chicago-Publican-Quality-Meats-PB-L/']



- 전체 50개로 잘 가져와 진것 같음

<br>

### 2.8 데이터프레임 생성


```python
import pandas as pd

data = {'Rank': rank, 'Menu': main_menu, 'Cafe': cafe_name, 'URL': url_add}
df = pd.DataFrame(data, columns=['Rank', 'Cafe', 'Menu', 'URL'])
df.tail()
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
      <th>Rank</th>
      <th>Cafe</th>
      <th>Menu</th>
      <th>URL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>45</th>
      <td>46</td>
      <td>Chickpea</td>
      <td>Kufta</td>
      <td>http://www.chicagomag.com/Chicago-Magazine/Nov...</td>
    </tr>
    <tr>
      <th>46</th>
      <td>47</td>
      <td>The Goddess and Grocer</td>
      <td>Debbie’s Egg Salad</td>
      <td>http://www.chicagomag.com/Chicago-Magazine/Nov...</td>
    </tr>
    <tr>
      <th>47</th>
      <td>48</td>
      <td>Zenwich</td>
      <td>Beef Curry</td>
      <td>http://www.chicagomag.com/Chicago-Magazine/Nov...</td>
    </tr>
    <tr>
      <th>48</th>
      <td>49</td>
      <td>Toni Patisserie</td>
      <td>Le Végétarien</td>
      <td>http://www.chicagomag.com/Chicago-Magazine/Nov...</td>
    </tr>
    <tr>
      <th>49</th>
      <td>50</td>
      <td>Phoebe’s Bakery</td>
      <td>The Gatsby</td>
      <td>http://www.chicagomag.com/Chicago-Magazine/Nov...</td>
    </tr>
  </tbody>
</table>
</div>



- 50개 자료에 대해 이름, 메뉴 , 상세페이지 URL까지 모두 정리 완료

<br>


```python
df.to_csv('./data/best_sandwiches_list.csv', sep =',', encoding = 'utf-8')
```

- csv 파일로 저장완료

<br>

## 3. 상세페이지
---
### 3.1 Data Load


```python
df = pd.read_csv('https://raw.githubusercontent.com/hmkim312/datas/main/chicagosandwiches/best_sandwiches_list.csv', index_col=0)
df.head()
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
      <th>Rank</th>
      <th>Cafe</th>
      <th>Menu</th>
      <th>URL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Old Oak Tap</td>
      <td>BLT</td>
      <td>https://www.chicagomag.com/Chicago-Magazine/No...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Au Cheval</td>
      <td>Fried Bologna</td>
      <td>https://www.chicagomag.com/Chicago-Magazine/No...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Xoco</td>
      <td>Woodland Mushroom</td>
      <td>https://www.chicagomag.com/Chicago-Magazine/No...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Al’s Deli</td>
      <td>Roast Beef</td>
      <td>https://www.chicagomag.com/Chicago-Magazine/No...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Publican Quality Meats</td>
      <td>PB&amp;L</td>
      <td>https://www.chicagomag.com/Chicago-Magazine/No...</td>
    </tr>
  </tbody>
</table>
</div>



- 데이터 불러옴

<br>

### 3.2 상세페이지 파싱


```python
html = urlopen(df['URL'][0])
soup_tmp = BeautifulSoup(html, 'html.parser')
soup_tmp
```

    <!DOCTYPE doctype html>
    
    <html lang="en">
    <head>
    <!-- Urbis magnitudo. Fabulas magnitudo. -->
    <meta charset="utf-8"/>
    <style>a.edit_from_site {display: none !important;}</style>
    <title>
      1. Old Oak Tap BLT |
      Chicago magazine
          |  November 2012
        </title>
    ...
    </script>
    <!--[if lt IE 9]>
    <script  type="text/javascript" language="JavaScript" src="/core/media/themes/Respond/js/respond.js?ver=1473876729"></script>
    <![endif]-->
    <script language="JavaScript" src="/core/media/js/base.js?ver=1473876728" type="text/javascript"></script>
    <script language="JavaScript" src="/core/media/themes/Respond/js/bootstrap.min.js?ver=1473876729" type="text/javascript"></script>
    <script language="JavaScript" src="//maps.googleapis.com/maps/api/js?v=3.exp&amp;sensor=false" type="text/javascript"></script>
    <script language="JavaScript" src="/theme_overrides/Respond/js/interstitial.js?ver=1524154906" type="text/javascript"></script>
    <script language="JavaScript" src="/theme_overrides/Respond/js/newsletter-subscribe.js?ver=1524850607" type="text/javascript"></script>
    <script language="JavaScript" src="/theme_overrides/Respond/js/RivistaGoogleDFP.js?ver=1447178886" type="text/javascript"></script>
    <!-- godengo-monitor --></body>
    </html>



- 하나의 페이지 파싱

<br>

<img src="https://user-images.githubusercontent.com/60168331/96877916-32de5600-14b5-11eb-963b-3a947c899818.png">

- p태그에 addy 클래스에 정보가 있는듯 하다

<br>

### 3.3 가격 및 주소 정보 가져오기


```python
price_tmp = soup_tmp.find('p', 'addy').get_text()
price_tmp
```
    '\n$10. 2109 W. Chicago Ave., 773-772-0406, theoldoaktap.com'



- 가격, 주소, 전화번호, 홈페이지 정보가 한번에 있음

<br>

```python
price_tmp = re.split('.,', price_tmp)[0]
price_tmp
```
    '\n$10. 2109 W. Chicago Ave'



- 미국은 주소 뒤에 .,가 붙으니, 그걸 기준으로 split하여 가격과 주소만 가져옴

<br>


```python
re.search('\$\d+\.(\d+)?', price_tmp).group()
```
    '$10.'



- 숫자로 시작하다가 꼭 .을 만나고 그 뒤에 숫자가 있을수도 있고 없을수도 있는 정규표현식으로 가격만 가져옴

<br>


```python
end = re.search('\$\d+\.(\d+)?', price_tmp).end()
price_tmp[end+1:]
```




    '2109 W. Chicago Ave'



- 가격이 끝나는 지점의 위치를 이용해서 그 뒤는 주소로 저장

<br>

### 3.4 3개에 시범삼아 적용


```python
price = []
address = []

for idx, row in df[:3].iterrows():
    html = urlopen(row['URL'])
    soup_tmp = BeautifulSoup(html, 'lxml')
    
    gettings = soup_tmp.find('p', 'addy').get_text()
    
    price_tmp = re.split('.,', gettings)[0]
    tmp = re.search('\$\d+\.(\d+)?', price_tmp).group()
    price.append(tmp)
    
    end = re.search('\$\d+\.(\d+)?', price_tmp).end()
    address.append(price_tmp[end+1:])
    
    print(row['Rank'])
```

    1
    2
    3



```python
price, address
```




    (['$10.', '$9.', '$9.50'],
     ['2109 W. Chicago Ave', '800 W. Randolph St', ' 445 N. Clark St'])



- 잘되는듯 하다.

<br>

### 3.5 50개 모두 돌리기


```python
import time
price = []
address = []

for idx, row in df.iterrows():
    html = urlopen(row['URL'])
    soup_tmp = BeautifulSoup(html, 'lxml')
    
    gettings = soup_tmp.find('p', 'addy').get_text()
    
    price_tmp = re.split('.,', gettings)[0]
    tmp = re.search('\$\d+\.(\d+)?', price_tmp).group()
    price.append(tmp)
    
    end = re.search('\$\d+\.(\d+)?', price_tmp).end()
    address.append(price_tmp[end+1:])
    
    time.sleep(0.5)
    print(row['Rank'], ' / ', '50')
```

    1  /  50
    2  /  50
    3  /  50
    ...
    47  /  50
    48  /  50
    49  /  50
    50  /  50



```python
len(price), len(address), len(df)
```

    (50, 50, 50)


- 잘 된듯 하다

<br>

### 3.6 데이터 프레임에 추가


```python
df['Price'] = price
df['Address'] = address

df = df.loc[:, ['Rank', 'Cafe', 'Menu', 'Price', 'Address']]
df.set_index('Rank', inplace = True)
df.head()
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
      <th>Cafe</th>
      <th>Menu</th>
      <th>Price</th>
      <th>Address</th>
    </tr>
    <tr>
      <th>Rank</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Old Oak Tap</td>
      <td>BLT</td>
      <td>$10.</td>
      <td>2109 W. Chicago Ave</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Au Cheval</td>
      <td>Fried Bologna</td>
      <td>$9.</td>
      <td>800 W. Randolph St</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Xoco</td>
      <td>Woodland Mushroom</td>
      <td>$9.50</td>
      <td>445 N. Clark St</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Al’s Deli</td>
      <td>Roast Beef</td>
      <td>$9.40</td>
      <td>914 Noyes St</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Publican Quality Meats</td>
      <td>PB&amp;L</td>
      <td>$10.</td>
      <td>825 W. Fulton Mkt</td>
    </tr>
  </tbody>
</table>
</div>



- 완료되었음. 

<br>

### 3.7 csv 저장


```python
df.to_csv('./data/best_sandwiches_list2.csv', sep =',', encoding = 'utf-8')
```
