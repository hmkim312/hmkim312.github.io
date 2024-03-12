---
title: Gemini API 사용해보기
author: HyunMin Kim
date: 2024-03-12 00:00:00 0000
categories: [Data Science, NLP]
tags: [LLM]
image: /assets/img/post/2024-03-12/thumbnail.png
---

최근 Google에서 Gemini를 발표했습니다. 이어서 구글 Bard를 Gemini로 이름을 바꿨는데요. OpenAI의 ChatGPT처럼 웹 버전과 API 버전을 모두 지원합니다. 이번 포스팅에서는 Gemini를 API로 사용해 보는 방법에 대해 소개하겠습니다. Google AI for Developers의 [Quickstarts](https://ai.google.dev/tutorials/python_quickstart?_gl=1*1ofwqyx*_up*MQ..*_ga*OTM0NDM2ODA5LjE3MDk3MzY2MzI.*_ga_P1DBVKWT6V*MTcwOTczNjYzMS4xLjAuMTcwOTczNjYzMS4wLjAuMA..)를 참고했습니다.


## 1. API키 발급받기
- Gemini는 *Google AI Dev*에서 API키를 발급 받을 수 있습니다.

![image](https://github.com/hmkim312/hmkim312/assets/60168331/c3de8cde-fd03-4127-b6bd-74e8dcd25591)

- 아래 링크로 접속하셔서 위 이미지에서 보이는 *Get Gemini API Key in Google AI Studio*를 클릭합니다.
- [Google AI Dev](https://ai.google.dev/?gad_source=1&gclid=CjwKCAiAxaCvBhBaEiwAvsLmWH8Z1_48C_ANSi7JdwpHyP1M7UyCQwljykHiHneeDeNzZkD0fqSPZBoCmeEQAvD_BwE)



![image](https://github.com/hmkim312/hmkim312/assets/60168331/04601719-e450-45c0-86f7-3a48e615417a)
- 왼쪽의 *Get API Key*를 클릭합니다.

![image](https://github.com/hmkim312/hmkim312/assets/60168331/9dabf607-278e-467e-9eaf-3aef3a34b96c)
- *Create API Key*를 눌러 API 키를 생성해주고, 해당 키를 잘 저장해 놓습니다.

![image](https://github.com/hmkim312/hmkim312/assets/60168331/12eb23c9-3caf-4fe4-acbe-128d2b7889f4)

- **참고** Google AI Studio에서 바로 Chatmode도 가능합니다.

## 2. Python 패키지 설치 및 import

```sh
pip install -q -U google-generativeai
```
- Python SDK를 설치해줍니다.


```python
import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

# 답변 정리용 Custom Module
def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
```

- Package를 import 합니다.

## 3. API Key 셋팅


```python
genai.configure(api_key="GOOGLE_API_KEY") #API를 입력해주세요!!!!!
```

- 앞에서 발급 받은 API Key를 설정합니다.

## 4. 모델 리스트


```python
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
```

    models/gemini-1.0-pro
    models/gemini-1.0-pro-001
    models/gemini-1.0-pro-latest
    models/gemini-1.0-pro-vision-latest
    models/gemini-pro
    models/gemini-pro-vision


- 사용 가능한 모델 리스트를 확인합니다.
- 현재는 gemini-1.0-pro와 gemini-pro, 그리고 vision 모델이 사용가능합니다.
- gemini는 현재 분당 60건의 요청까지 무료입니다. (2024.03.12 기준)
- 자세한 요금은 [여기](https://ai.google.dev/pricing?hl=ko)애서 확인가능합니다.

## 5. 텍스트 생성


```python
%%time
model = genai.GenerativeModel("gemini-pro")
response = model.generate_content("손흥민이 누구야?")
```

    CPU times: user 7.77 ms, sys: 5.44 ms, total: 13.2 ms
    Wall time: 9.81 s


- gemini-pro 모델을 불러오고, 손흥민이 누군지 물어보았습니다.
- 약 6초 가량 소요됐습니다. 답변이 빠른편은 아니지만, 생성된 답변을 보면 엄청 느린것 같지도 않습니다.


```python
to_markdown(response.text)
```




> 손흥민은 대한민국 출신 축구 선수로, 포워드로 선수로 활약하고 있습니다.
> 
> **주요 성취 및 기록:**
> 
> * **잉글랜드 프리미어리그 득점왕 (2021-22 시즌)**
> * **UEFA 챔피언스리그 득점왕 (2022-23 시즌)**
> * **아시아 축구 연맹 올해의 남자 선수상 (2014, 2015, 2017, 2019, 2020)**
> * **K리그 득점왕 (2008)**
> * **올림픽 동메달 (2012)**
> * **아시안컵 우승 (2015)**
> * **잉글랜드 프리미어리그 구단 최다 해외 선수 득점 기록**
> * **토트넘 홋스퍼 구단 역대 최다 해외 선수 득점 기록**
> 
> 현재 토트넘 홋스퍼와 대한민국 국가대표팀에서 활약하고 있습니다. 그는 빠른 발놀림, 뛰어난 기회 만들기 능력, 강력한 슈팅 능력으로 유명합니다.



- gemini의 대답을 확인해보겠습니다.


```python
response.prompt_feedback
```




    safety_ratings {
      category: HARM_CATEGORY_SEXUALLY_EXPLICIT
      probability: NEGLIGIBLE
    }
    safety_ratings {
      category: HARM_CATEGORY_HATE_SPEECH
      probability: LOW
    }
    safety_ratings {
      category: HARM_CATEGORY_HARASSMENT
      probability: NEGLIGIBLE
    }
    safety_ratings {
      category: HARM_CATEGORY_DANGEROUS_CONTENT
      probability: NEGLIGIBLE
    }



- Gemini API는 성적, 혐오, 괴롭힘, 위험한 내용에 대해서는 응답하지 않는것을 *prompt_feedback*으로 통해 확인할 수 있습니다.


```python
response.candidates
```




    [index: 0
    content {
      parts {
        text: "손흥민은 대한민국 출신 축구 선수로, 포워드로 선수로 활약하고 있습니다.\n\n**주요 성취 및 기록:**\n\n* **잉글랜드 프리미어리그 득점왕 (2021-22 시즌)**\n* **UEFA 챔피언스리그 득점왕 (2022-23 시즌)**\n* **아시아 축구 연맹 올해의 남자 선수상 (2014, 2015, 2017, 2019, 2020)**\n* **K리그 득점왕 (2008)**\n* **올림픽 동메달 (2012)**\n* **아시안컵 우승 (2015)**\n* **잉글랜드 프리미어리그 구단 최다 해외 선수 득점 기록**\n* **토트넘 홋스퍼 구단 역대 최다 해외 선수 득점 기록**\n\n현재 토트넘 홋스퍼와 대한민국 국가대표팀에서 활약하고 있습니다. 그는 빠른 발놀림, 뛰어난 기회 만들기 능력, 강력한 슈팅 능력으로 유명합니다."
      }
      role: "model"
    }
    finish_reason: STOP
    safety_ratings {
      category: HARM_CATEGORY_SEXUALLY_EXPLICIT
      probability: NEGLIGIBLE
    }
    safety_ratings {
      category: HARM_CATEGORY_HATE_SPEECH
      probability: NEGLIGIBLE
    }
    safety_ratings {
      category: HARM_CATEGORY_HARASSMENT
      probability: NEGLIGIBLE
    }
    safety_ratings {
      category: HARM_CATEGORY_DANGEROUS_CONTENT
      probability: NEGLIGIBLE
    }
    ]



- 여러 응답 후보를 보려면 *candidates*로 확인 가능합니다.


```python
%%time
response = model.generate_content("손흥민이 누구야?", stream=True)
for chunk in response:
    print(chunk.text)
    print("_"*80)
```

    손흥민은 대한민국의 축구 선수로, 현재 프리미어 리
    ________________________________________________________________________________
    그 클럽인 토트넘 홋스퍼에서 공격수로 활약하고 있습니다.
    
    **주요 이력:**
    
    * 1
    ________________________________________________________________________________
    992년 7월 8일 대한민국 춘천 출생
    * FC 서울에서 데뷔하여 K리그 클래식에서 3회 우승
    * 2015년 레버쿠젠으로 이적하여 분데스리가에서 인상적인 성과를
    ________________________________________________________________________________
     거둠
    * 2017년 토트넘 홋스퍼로 이적하여 잉글랜드 프리미어 리그에서 득점 순위 3위를 차지
    * 2020년과 2022년 프리미어 리그 골든 부트 수상
    * 2022년 FIFA 월드컵에서 아시아 역사상 처음으로 월드컵 조별 리그에서 해트트릭을 달성
    * 대한민국 국가대표팀 주장이며, 대표팀 통산 111경기 출
    ________________________________________________________________________________
    장 36골 기록
    * 타임지에서 "세계에서 가장 영향력 있는 100인"에 선정(2022)
    
    손흥민은 빠른 속도, 날카로운 슈팅 능력, 탁월한 기술로 유명하며, 현재 세계 최고의 공격수 중 한 명으로 인정받고 있습니다.
    ________________________________________________________________________________
    CPU times: user 11.9 ms, sys: 12.6 ms, total: 24.5 ms
    Wall time: 5.9 s


- 기본적으로 모델은 응답이 모두 생성된 후 Response해줍니다. 
- 하지만 응답이 생성되는 동안 스트리밍 할 수도 있으며, 모델은 응답이 생성되는 즉시 청크를 Return 해줍니다.
- 이러한 스트리밍 기능은 stream 옵션을 통해 사용가능합니다.

## 6. 이미지로 텍스트 생성


```python
!curl -o image.jpg https://t0.gstatic.com/licensed-image?q=tbn:ANd9GcQ_Kevbk21QBRy-PgB4kQpS79brbmmEG7m3VOTShAn4PecDU5H5UxrJxE3Dw1JiaG17V88QIol19-3TM2wCHw
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  405k  100  405k    0     0   862k      0 --:--:-- --:--:-- --:--:--  861k


- 샘플로 사용될 이미지를 다운 받습니다.


```python
from IPython.display import Image
from IPython.core.display import HTML
img= Image('image.jpg')
```

![image](https://github.com/hmkim312/datas/assets/60168331/35949368-5728-4024-8c18-1bc2cfbff7d0)
- Vision Pro 모델을 사용해서 해당 이미지를 설명하게 해보겠습니다.


```python
model = genai.GenerativeModel('gemini-pro-vision')
response = model.generate_content(contents=img)
to_markdown(response.text)
```




>  These are meal prep containers with chicken, brown rice, broccoli, and carrots.



- 이미지와 관련된 내용을 설명해줍니다.


```python
model = genai.GenerativeModel('gemini-pro-vision')
response = model.generate_content([img, "이미지를 한국어로 설명해줘"])
to_markdown(response.text)
```




>  이미지는 도시락을 담은 유리 용기를 보여줍니다. 도시락에는 닭고기, 브로콜리, 당근, 밥이 들어 있습니다. 용기 옆에는 젓가락이 있습니다.



- 프롬프트에 텍스트와 이미지를 모두 넣으려면, 문자열과 이미지가 포함된 list를 넣으면 됩니다.

## 7. 대화 모드


```python
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])
chat
```




    ChatSession(
        model=genai.GenerativeModel(
            model_name='models/gemini-pro',
            generation_config={},
            safety_settings={},
            tools=None,
        ),
        history=[]
    )




```python
response = chat.send_message("컴퓨터 작동법을 어린아이에게 설명하듯이 알려줘")
to_markdown(response.text)
```




> 컴퓨터는 작은 친구처럼 생각해볼 수 있는 거야! 컴퓨터도 우리처럼 뇌와 몸이 있어.
> 
> **뇌 (CPU):**
> 뇌는 컴퓨터에서 가장 중요한 부분이야. 우리가 컴퓨터에 게임을 하거나 인터넷을 찾을 때 모든 생각을 하는 거야.
> 
> **몸 (하드 드라이브):**
> 몸은 컴퓨터가 게임, 노래, 사진을 기억하는 곳이야. 就像我们把东西放在书包里一样, 컴퓨터는 기억하려는 모든 것을 하드 드라이브에 넣어두는 거야.
> 
> **입 (USB 포트):**
> 입은 컴퓨터가 외부 세계와 연결되는 거야. 마치 우리가 입으로 음식을 먹는 것처럼, 컴퓨터는 USB 포트를 통해 프린터나 마우스와 같은 다른 기기를 연결할 수 있어.
> 
> **눈 (모니터):**
> 컴퓨터의 눈을 모니터라고 해. 모니터를 통해 컴퓨터가 생각한 내용을 볼 수 있어. 마치 우리가 책을 읽듯이, 모니터를 통해 게임, 인터넷, 사진을 볼 수 있는 거야.
> 
> **키보드와 마우스:**
> 키보드와 마우스는 우리가 컴퓨터에게 명령을 내리는 도구야. 마치 우리가 말하고 몸을 움직여 세상과 소통하듯이, 키보드와 마우스를 통해 컴퓨터에 무엇을 원하는지 말할 수 있어.
> 
> 이렇게 컴퓨터는 작은 친구처럼 생각하면서 뇌, 몸, 입, 눈, 키보드와 마우스를 가지고 있어. 이 모든 부분이 함께 작동해서 우리가 게임을 하고, 인터넷을 찾고, 사진을 저장할 수 있게 해주는 거야!



- Gemini를 통해 자유로운 대화를 할 수 있습니다.


```python
response = chat.send_message('램에 대해서 자세히 알려줄래', stream=True)
for chunk in response:
    print(chunk.text)
    print("_"*80)
```

    **RAM(Random Access Memory)**
    
    RAM은 컴퓨터의 주
    ________________________________________________________________________________
    요 메모리 유형으로, 컴퓨터가 현재 사용 중인 데이터와 명령어를 저장하는 데 사용됩니다. 휘발
    ________________________________________________________________________________
    성 메모리이므로 컴퓨터가 꺼지면 저장된 데이터가 손실됩니다.
    
    **RAM의 작동 원리**
    
    RAM은 메모리 셀 배열로 구성되어 있으며, 각 셀은 1비트(0 또는 1)의 데이터를
    ________________________________________________________________________________
     저장할 수 있습니다. 이러한 셀은 주소 라인과 데이터 라인을 통해 액세스됩니다.
    
    컴퓨터가 데이터를 RAM에 쓰려면 다음 단계를 수행합니다.
    
    1. 컴퓨터는 데이터를 저장할 메모리 주소를 지정합니다.
    2. 데이터 라인에는 저장할 데이터가 로드됩니다.
    3. 주소 라인은 지정된 주소를 가리키도록 설정됩니다.
    4. 제어 신호가 RAM에 데이터를 저장하도록 지시합니다.
    
    데이터를 RAM
    ________________________________________________________________________________
    에서 읽으려면 다음 단계를 수행합니다.
    
    1. 컴퓨터는 읽을 메모리 주소를 지정합니다.
    2. 주소 라인은 지정된 주소를 가리키도록 설정됩니다.
    3. 제어 신호가 RAM에 데이터를 읽으라고 지시합니다.
    4. 데이터 라인에는 읽은 데이터가 로드됩니다.
    
    **RAM 유형**
    
    RAM에는 여러 유형이 있으며, 가장 일반적인 유형은 다음과 같습니다.
    
    * **DRAM(동적 RAM):** 가장 일반적인 RAM 유형이며, 주기적으로 갱신해야 합니다.
    * **SRAM(정적 RAM):** DRAM보다 빠르고 비싸지만 갱신이 필요하지 않습니다.
    * **DDR RAM(Double Data Rate RAM):** 데이터를 두 번 전송하여 속도를 향상시킨 DRAM 유형입니다.
    
    **RAM의 용도**
    
    RAM은 다음과 같은 목적으로 사용됩니다.
    
    * 운영 체제 및 프로그램 로드
    * 실행 중인 프로그램에 대한 데이터 저장
    * 그래픽 및 오디오 데이터 임시 저장
    
    **RAM 용량**
    
    RAM
    ________________________________________________________________________________
     용량은 컴퓨터의 성능에 중요한 영향을 미칩니다. RAM 용량이 클수록 컴퓨터가 한 번에 더 많은 데이터와 명령어를 처리할 수 있습니다. 일반적으로 게임, 그래픽 작업, 다중 작업을 위해서는 더 많은 RAM이 필요합니다.
    
    **RAM 업그레이드**
    
    RAM 용량이 부족하면 컴퓨터 성능이 저하될 수 있습니다. 이러한 경우 RAM을 업그레이드하는 것이 좋습니다. 대부분의 컴퓨터는 RAM을 업그레이드할 수 있도록 설계되어 있습니다.
    ________________________________________________________________________________


- 계속해서 대화를 이어갈 수 있으며, stream 옵션도 가능합니다.


```python
chat.history
```




    [parts {
       text: "컴퓨터 작동법을 어린아이에게 설명하듯이 알려줘"
     }
     role: "user",
     parts {
       text: "컴퓨터는 작은 친구처럼 생각해볼 수 있는 거야! 컴퓨터도 우리처럼 뇌와 몸이 있어.\n\n**뇌 (CPU):**\n뇌는 컴퓨터에서 가장 중요한 부분이야. 우리가 컴퓨터에 게임을 하거나 인터넷을 찾을 때 모든 생각을 하는 거야.\n\n**몸 (하드 드라이브):**\n몸은 컴퓨터가 게임, 노래, 사진을 기억하는 곳이야. 就像我们把东西放在书包里一样, 컴퓨터는 기억하려는 모든 것을 하드 드라이브에 넣어두는 거야.\n\n**입 (USB 포트):**\n입은 컴퓨터가 외부 세계와 연결되는 거야. 마치 우리가 입으로 음식을 먹는 것처럼, 컴퓨터는 USB 포트를 통해 프린터나 마우스와 같은 다른 기기를 연결할 수 있어.\n\n**눈 (모니터):**\n컴퓨터의 눈을 모니터라고 해. 모니터를 통해 컴퓨터가 생각한 내용을 볼 수 있어. 마치 우리가 책을 읽듯이, 모니터를 통해 게임, 인터넷, 사진을 볼 수 있는 거야.\n\n**키보드와 마우스:**\n키보드와 마우스는 우리가 컴퓨터에게 명령을 내리는 도구야. 마치 우리가 말하고 몸을 움직여 세상과 소통하듯이, 키보드와 마우스를 통해 컴퓨터에 무엇을 원하는지 말할 수 있어.\n\n이렇게 컴퓨터는 작은 친구처럼 생각하면서 뇌, 몸, 입, 눈, 키보드와 마우스를 가지고 있어. 이 모든 부분이 함께 작동해서 우리가 게임을 하고, 인터넷을 찾고, 사진을 저장할 수 있게 해주는 거야!"
     }
     role: "model"]



- 대화 기록도 볼 수 있습니다.


```python
for message in chat.history:
    display(to_markdown(f'**{message.role}**: {message.parts[0].text}'))
```


> **user**: 컴퓨터 작동법을 어린아이에게 설명하듯이 알려줘



> **model**: **컴퓨터를 어린이에게 설명하는 방법:**
> 
> 컴퓨터는 너가 놀고 일할 수 있는 특별한 기계야.
> 
> **컴퓨터의 구성 부품은 이렇게 생겼어:**
> 
> * **모니터:** 이건 컴퓨터 화면이야. 너가 게임을 하거나 영상을 볼 수 있어.
> * **마우스:** 이건 컴퓨터를 조작하는 데 쓰는 작은 기계야.
> * **키보드:** 이건 컴퓨터에 글을 입력할 때 쓰는 거야.
> * **타워:** 이건 컴퓨터의 뇌야. 모든 것을 처리해 줘.
> 
> **컴퓨터가 하는 일은 다음과 같아:**
> 
> * **게임 플레이:** 너가 좋아하는 게임을 컴퓨터에서 할 수 있어.
> * **영상 시청:** 컴퓨터로 웃긴 영상이나 재미있는 영화를 볼 수 있어.
> * **학습:** 컴퓨터로 새로운 것을 배우고 교육용 게임을 할 수 있어.
> * **글쓰기:** 컴퓨터로 숙제를 하거나 이야기를 쓸 수 있어.
> * **그리기:** 컴퓨터로 그림을 그리거나 색칠을 할 수 있어.
> 
> **컴퓨터를 사용하는 방법은 이렇게 간단해:**
> 
> * 마우스를 움직여 화면에 있는 물건을 가리켜 봐.
> * 마우스를 클릭해서 가리킨 물건을 열거나 선택해 봐.
> * 키보드를 사용해서 글을 입력해 봐.
> * 게임을 하거나 다른 프로그램을 실행해 봐.
> 
> 컴퓨터는 너에게 많은 재미를 선사해 줄 수 있는 멋진 도구야. 안전하게 사용하고 책임감 있게 사용하는 것을 잊지 마렴!



> **user**: 램에 대해서 자세히 알려줄래



> **model**: **RAM(Random Access Memory)**
> 
> RAM은 컴퓨터의 주요 메모리 유형으로, 컴퓨터가 현재 사용 중인 데이터와 명령어를 저장하는 데 사용됩니다. 휘발성 메모리이므로 컴퓨터가 꺼지면 저장된 데이터가 손실됩니다.
> 
> **RAM의 작동 원리**
> 
> RAM은 메모리 셀 배열로 구성되어 있으며, 각 셀은 1비트(0 또는 1)의 데이터를 저장할 수 있습니다. 이러한 셀은 주소 라인과 데이터 라인을 통해 액세스됩니다.
> 
> 컴퓨터가 데이터를 RAM에 쓰려면 다음 단계를 수행합니다.
> 
> 1. 컴퓨터는 데이터를 저장할 메모리 주소를 지정합니다.
> 2. 데이터 라인에는 저장할 데이터가 로드됩니다.
> 3. 주소 라인은 지정된 주소를 가리키도록 설정됩니다.
> 4. 제어 신호가 RAM에 데이터를 저장하도록 지시합니다.
> 
> 데이터를 RAM에서 읽으려면 다음 단계를 수행합니다.
> 
> 1. 컴퓨터는 읽을 메모리 주소를 지정합니다.
> 2. 주소 라인은 지정된 주소를 가리키도록 설정됩니다.
> 3. 제어 신호가 RAM에 데이터를 읽으라고 지시합니다.
> 4. 데이터 라인에는 읽은 데이터가 로드됩니다.
> 
> **RAM 유형**
> 
> RAM에는 여러 유형이 있으며, 가장 일반적인 유형은 다음과 같습니다.
> 
> * **DRAM(동적 RAM):** 가장 일반적인 RAM 유형이며, 주기적으로 갱신해야 합니다.
> * **SRAM(정적 RAM):** DRAM보다 빠르고 비싸지만 갱신이 필요하지 않습니다.
> * **DDR RAM(Double Data Rate RAM):** 데이터를 두 번 전송하여 속도를 향상시킨 DRAM 유형입니다.
> 
> **RAM의 용도**
> 
> RAM은 다음과 같은 목적으로 사용됩니다.
> 
> * 운영 체제 및 프로그램 로드
> * 실행 중인 프로그램에 대한 데이터 저장
> * 그래픽 및 오디오 데이터 임시 저장
> 
> **RAM 용량**
> 
> RAM 용량은 컴퓨터의 성능에 중요한 영향을 미칩니다. RAM 용량이 클수록 컴퓨터가 한 번에 더 많은 데이터와 명령어를 처리할 수 있습니다. 일반적으로 게임, 그래픽 작업, 다중 작업을 위해서는 더 많은 RAM이 필요합니다.
> 
> **RAM 업그레이드**
> 
> RAM 용량이 부족하면 컴퓨터 성능이 저하될 수 있습니다. 이러한 경우 RAM을 업그레이드하는 것이 좋습니다. 대부분의 컴퓨터는 RAM을 업그레이드할 수 있도록 설계되어 있습니다.


- 대화한 내용을 user와 model로 나누어서 볼 수 있습니다.

## 8. 임베딩
- 임베딩은 정보를 배열의 부동 소수점 숫자 목록으로 표현하는 데 사용되는 기술입니다. 
- Gemini를 사용하면 텍스트(단어, 문장 및 텍스트 블록)를 벡터화된 형식으로 표현할 수 있으므로 임베딩을 더 쉽게 비교하고 대조할 수 있습니다.
- 임베딩은 정보를 배열의 부동 소수점 숫자 목록으로 표현하는 데 사용되는 기술입니다. 
- Gemini를 사용하면 텍스트(단어, 문장 및 텍스트 블록)를 벡터화된 형식으로 표현할 수 있으므로 임베딩을 더 쉽게 비교하고 대조할 수 있습니다


```python
result = genai.embed_content(
    model="models/embedding-001",
    content="손흥민이 누구야",
    task_type="retrieval_document",
    title="Embedding of single string")

# 1 input > 1 vector output
print(str(result['embedding'])[:50], '... TRIMMED]')
```

    [0.03795641, -0.006939522, -0.014510555, -0.004150 ... TRIMMED]


- Title은 retrieval_document 유형에서 사용 가능합니다.


```python
result = genai.embed_content(
    model="models/embedding-001",
    content=[
      '손흥민이 누구야?',
      '박지성은 누구야?',
      'Gemini는 뭐야?'],
    task_type="retrieval_document",
    title="Embedding of list of strings")

# A list of inputs > A list of vectors output
for v in result['embedding']:
    print(str(v)[:50], '... TRIMMED ...')
```

    [0.039229624, -0.0030471336, -0.014464416, 0.01432 ... TRIMMED ...
    [0.039229624, -0.0030471336, -0.014464416, 0.01432 ... TRIMMED ...
    [0.048528966, 0.0055916472, -0.016637022, 0.010156 ... TRIMMED ...


- content를 List 형태로 입력하면 배치형태로 사용 가능합니다.


```python
chat.history
```




    [parts {
       text: "컴퓨터 작동법을 어린아이에게 설명하듯이 알려줘"
     }
     role: "user",
     parts {
       text: "**컴퓨터를 어린이에게 설명하는 방법:**\n\n컴퓨터는 너가 놀고 일할 수 있는 특별한 기계야.\n\n**컴퓨터의 구성 부품은 이렇게 생겼어:**\n\n* **모니터:** 이건 컴퓨터 화면이야. 너가 게임을 하거나 영상을 볼 수 있어.\n* **마우스:** 이건 컴퓨터를 조작하는 데 쓰는 작은 기계야.\n* **키보드:** 이건 컴퓨터에 글을 입력할 때 쓰는 거야.\n* **타워:** 이건 컴퓨터의 뇌야. 모든 것을 처리해 줘.\n\n**컴퓨터가 하는 일은 다음과 같아:**\n\n* **게임 플레이:** 너가 좋아하는 게임을 컴퓨터에서 할 수 있어.\n* **영상 시청:** 컴퓨터로 웃긴 영상이나 재미있는 영화를 볼 수 있어.\n* **학습:** 컴퓨터로 새로운 것을 배우고 교육용 게임을 할 수 있어.\n* **글쓰기:** 컴퓨터로 숙제를 하거나 이야기를 쓸 수 있어.\n* **그리기:** 컴퓨터로 그림을 그리거나 색칠을 할 수 있어.\n\n**컴퓨터를 사용하는 방법은 이렇게 간단해:**\n\n* 마우스를 움직여 화면에 있는 물건을 가리켜 봐.\n* 마우스를 클릭해서 가리킨 물건을 열거나 선택해 봐.\n* 키보드를 사용해서 글을 입력해 봐.\n* 게임을 하거나 다른 프로그램을 실행해 봐.\n\n컴퓨터는 너에게 많은 재미를 선사해 줄 수 있는 멋진 도구야. 안전하게 사용하고 책임감 있게 사용하는 것을 잊지 마렴!"
     }
     role: "model",
     parts {
       text: "램에 대해서 자세히 알려줄래"
     }
     role: "user",
     parts {
       text: "**RAM(Random Access Memory)**\n\nRAM은 컴퓨터의 주요 메모리 유형으로, 컴퓨터가 현재 사용 중인 데이터와 명령어를 저장하는 데 사용됩니다. 휘발성 메모리이므로 컴퓨터가 꺼지면 저장된 데이터가 손실됩니다.\n\n**RAM의 작동 원리**\n\nRAM은 메모리 셀 배열로 구성되어 있으며, 각 셀은 1비트(0 또는 1)의 데이터를 저장할 수 있습니다. 이러한 셀은 주소 라인과 데이터 라인을 통해 액세스됩니다.\n\n컴퓨터가 데이터를 RAM에 쓰려면 다음 단계를 수행합니다.\n\n1. 컴퓨터는 데이터를 저장할 메모리 주소를 지정합니다.\n2. 데이터 라인에는 저장할 데이터가 로드됩니다.\n3. 주소 라인은 지정된 주소를 가리키도록 설정됩니다.\n4. 제어 신호가 RAM에 데이터를 저장하도록 지시합니다.\n\n데이터를 RAM에서 읽으려면 다음 단계를 수행합니다.\n\n1. 컴퓨터는 읽을 메모리 주소를 지정합니다.\n2. 주소 라인은 지정된 주소를 가리키도록 설정됩니다.\n3. 제어 신호가 RAM에 데이터를 읽으라고 지시합니다.\n4. 데이터 라인에는 읽은 데이터가 로드됩니다.\n\n**RAM 유형**\n\nRAM에는 여러 유형이 있으며, 가장 일반적인 유형은 다음과 같습니다.\n\n* **DRAM(동적 RAM):** 가장 일반적인 RAM 유형이며, 주기적으로 갱신해야 합니다.\n* **SRAM(정적 RAM):** DRAM보다 빠르고 비싸지만 갱신이 필요하지 않습니다.\n* **DDR RAM(Double Data Rate RAM):** 데이터를 두 번 전송하여 속도를 향상시킨 DRAM 유형입니다.\n\n**RAM의 용도**\n\nRAM은 다음과 같은 목적으로 사용됩니다.\n\n* 운영 체제 및 프로그램 로드\n* 실행 중인 프로그램에 대한 데이터 저장\n* 그래픽 및 오디오 데이터 임시 저장\n\n**RAM 용량**\n\nRAM 용량은 컴퓨터의 성능에 중요한 영향을 미칩니다. RAM 용량이 클수록 컴퓨터가 한 번에 더 많은 데이터와 명령어를 처리할 수 있습니다. 일반적으로 게임, 그래픽 작업, 다중 작업을 위해서는 더 많은 RAM이 필요합니다.\n\n**RAM 업그레이드**\n\nRAM 용량이 부족하면 컴퓨터 성능이 저하될 수 있습니다. 이러한 경우 RAM을 업그레이드하는 것이 좋습니다. 대부분의 컴퓨터는 RAM을 업그레이드할 수 있도록 설계되어 있습니다."
     }
     role: "model"]



- 앞에서 사용한 chat의 대화내용도 임베딩이 가능합니다.


```python
result = genai.embed_content(
    model = 'models/embedding-001',
    content = chat.history)

# 1 input > 1 vector output
for i,v in enumerate(result['embedding']):
    print(str(v)[:50], '... TRIMMED...')
```

    [0.058809474, -0.041524325, -0.01679906, 0.0016449 ... TRIMMED...
    [0.041108645, -0.043516986, -0.003114645, 0.015857 ... TRIMMED...
    [0.05925582, -0.04323863, -0.020118687, 0.00013299 ... TRIMMED...
    [0.05087915, -0.022734836, -0.024366891, 0.0146824 ... TRIMMED...


- genai.embed_content는 텍스트나, List를 지원합니다. 
- 임베딩은 Content 유형을 기준으로 실행되며, Content는 텍스트 뿐만 아니라 그 외의 것을 이야기합니다.
- 즉, Content 객체는 멀티모달 이며, 임베딩 API를 멀티모달 임베딩으로 확장 가능성을 의미합니다.
- 하지만 아직 임베딩은 텍스트 형태만 지원합니다.


```python
model = genai.GenerativeModel('gemini-pro')

messages = [
    {'role':'user',
     'parts': ["손흥민이 누구야"]}
]
response = model.generate_content(messages)

to_markdown(response.text)
```




> 손흥민은 대한민국 축구 선수로, 현재 잉글랜드 프리미어리그의 토트넘 홋스퍼에서 공격수로 활약하고 있습니다.
> 
> **약력:**
> 
> * 1992년 7월 8일 서울에서 태어남
> * FC 서울 유소년팀에서 축구를 시작
> * 2010년 FC 서울에서 프로 데뷔
> * 2013년 독일 분데스리가의 함부르크 SV로 이적
> * 2015년 독일 분데스리가의 바이어 레버쿠젠으로 이적
> * 2015년 8월 잉글랜드 프리미어리그의 토트넘 홋스퍼로 이적
> * 2020년 토트넘의 역대 최다 골 기록을 경신
> 
> **국가대표팀:**
> 
> * 2010년 대한민국 U-20 국가대표팀 소속으로 FIFA U-20 월드컵 참가
> * 2011년 대한민국 국가대표팀에 데뷔
> * 2014, 2018, 2022년 FIFA 월드컵에 참가
> * 2018년 아시안 게임에서 금메달을 획득
> * 대한민국 국가대표팀의 역대 최다 골 기록 보유자
> 
> **주요 수상 내역:**
> 
> * 프리미어리그 득점왕 (2021-22)
> * PFA 올해의 선수상 (2021-22)
> * AFC 아시아 올해의 선수상 (2015, 2017, 2018, 2019)
> * KFA 올해의 선수상 (2013, 2014, 2017, 2019, 2020, 2021, 2022)
> 
> 손흥민은 대한민국의 축구 역사상 가장 성공한 선수 중 한 명으로, 경기력, 속도, 마무리 능력으로 유명합니다. 그는 전 세계적인 인정을 받으며, 프리미어리그와 대한민국 국가대표팀의 스타 선수로 자리 잡았습니다.



## 9. 멀티 턴 대화


```python
messages.append({'role':'model',
                 'parts':[response.text]})

messages.append({'role':'user',
                 'parts':["주요수상내역에 대해 더 자세히 알려줄래"]})

response = model.generate_content(messages)

to_markdown(response.text)
```




> **주요 수상 내역:**
> 
> **프리미어리그 득점왕 (2021-22)**
> 
> * 손흥민은 2021-22시즌에 리버풀의 모하메드 살라와 함께 23골을 기록하며 프리미어리그 득점왕에 올랐습니다. 이는 아시아 선수로서는 이례적인 성과였으며, 프리미어리그 역사상 아시아인 득점왕은 손흥민이 처음입니다.
> 
> **PFA 올해의 선수상 (2021-22)**
> 
> * PFA 올해의 선수상은 선수들이 선정하는 상으로, 손흥민은 2021-22시즌에 프리미어리그에서 가장 뛰어난 선수로 선정되었습니다. 그는 에런 램지, 박지성에 이어 이 상을 받은 세 번째 아시아 선수입니다.
> 
> **AFC 아시아 올해의 선수상 (2015, 2017, 2018, 2019)**
> 
> * AFC 아시아 올해의 선수상은 아시아 축구 연맹(AFC)이 선정하는 상으로, 손흥민은 2015년, 2017년, 2018년, 2019년에 4번이나 이 상을 수상했습니다. 그는 이 상을 가장 많이 수상한 선수이며, 대한민국 선수로서는 최초로 수상한 선수입니다.
> 
> **KFA 올해의 선수상 (2013, 2014, 2017, 2019, 2020, 2021, 2022)**
> 
> * KFA 올해의 선수상은 대한민국 축구 협회(KFA)가 선정하는 상으로, 손흥민은 2013년부터 2022년까지 7번이나 이 상을 수상했습니다. 그는 박지성에 이어 이 상을 가장 많이 수상한 선수입니다.
> 
> 이러한 수상 외에도 손흥민은 다음과 같은 수상도 받았습니다.
> 
> * AFC 챔피언스리그 MVP (2011)
> * 독일 분데스리가 월간 MVP (2013년 12월, 2014년 3월)
> * 토트넘 홋스퍼 올해의 선수상 (2016-17, 2018-19, 2020-21, 2021-22)
> * FIFA 푸스카시상 후보 (2015, 2020)
> 
> 손흥민의 수상 내역은 그의 탁월한 축구 기술과 아시아 축구의 발전에 기여한 공로를 인정받은 것입니다.



- glm.Content 개체 목록을 전달하면 멀티턴 채팅으로 처리됩니다
- 멀티턴 대화는 API에 상태가 저장되지 않으므로 요청을 보낼때 항상 전체 대화 내역을 보내야 합니다.
