---
title: Ollama와 Python 라이브러리를 이용하여 LLaMa2를 로컬에서 사용하기 
author: HyunMin Kim
date: 2024-02-13 00:00:00 0000
categories: [Data Science, NLP]
tags: [LLM]
image: /assets/img/post/2024-02-13/thumbnail.png
---

최근 생성형 AI가 굉장히 많은 주목을 받고 있다. OpenAI, Google, Meta 등 거대 기업들을 필두로 생성형 AI는 빠른 발전을 이루고 있는데요. 이러한 생성형 AI를 사용자들이 더 손쉽게 사용하게 도와주는 OllaMa와 Python 라이브러리가 발표되어 간단하게 알아 보았다.

이번 포스트에서는 아래의 2가지를 중점으로 알아본다.
1. Terminal 환경에서 Ollama 사용하기 (w. LLaMa2)
2. Ollama Python 라이브러리 사용하기


### 1. Terminal 환경에서 Ollama 사용하기 (w. LLaMa2)
- https://ollama.com/download 에서 자신의 OS에 맞는 설치 파일을 다운 받아 설치한다.
- e.g) 리눅스
```shell
$curl -fsSL https://ollama.com/install.sh | sh
```


<img width="1166" alt="스크린샷 2024-02-13 오후 10 39 13" src="https://github.com/hmkim312/datas/assets/60168331/1c684f09-8c01-432c-b7a2-0041a577944b">

```shell
$ollama run llama2
```
- 설치가 완료 후 `ollama run llama2` 명령어룰 통해 llama2를 실행시켜 채팅을 진행해본다.
- 로컬 컴퓨터는 RTX4070으로 VRAM은 12GB이며 LLaMa2와 채팅의 답변 속도는 꽤 빠른편이었다.
- 다만, 성능은 ChatGPT와는 비교 불가이다. 로컬에서 사용하는 sLLM의 한계 인 것 같다.
- LLaMa2뿐만 아니라 ollama가 지원하는 다른 모델도 사용 할 수 있다. [Ollama Library](https://ollama.com/libraryhttps://ollama.com/library)

### 2. Ollama Python 라이브러리 사용하기
```shell
$pip install ollama
```
- ollama python 라이브러리를 설치한다.
- 참고) 1번에서 ollama를 설치 후 ollama python 라이브러리를 설치해야 잘 작동한다.

#### 2-1 기본 사용

```python
import ollama
response = ollama.chat(model='llama2', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])
```

    
    The sky appears blue because of a phenomenon called Rayleigh scattering. When sunlight enters Earth's atmosphere, it encounters tiny molecules of gases such as nitrogen and oxygen. These molecules scatter the light in all directions, but they scatter shorter (blue) wavelengths more than longer (red) wavelengths. This is known as Rayleigh scattering.
    
    As a result of this scattering, the blue light is distributed throughout the atmosphere, giving the sky its blue appearance. The red light, on the other hand, passes through the atmosphere with little scattering, which is why we can see the sun's red light from the ground.
    
    The amount of scattering that occurs depends on the size of the particles in the atmosphere and the wavelength of the light. For example, during sunrise and sunset, the sky may appear more orange or red because the sun's rays are passing through a greater distance in the atmosphere, causing more scattering of the blue light.
    
    So, to summarize, the sky appears blue because of the way light interacts with the tiny molecules of gases in Earth's atmosphere, resulting in Rayleigh scattering that distributes blue light throughout the atmosphere.


- python 환경에서 ollama를 import하여 로컬에 설치되어있는 llama2를 실행하여 채팅을 합니다.
- Langchain의 사용법과 굉장히 비슷하여 익숙합니다.

#### 2-2 스트리밍 사용

```python
import ollama

stream = ollama.chat(
    model='llama2',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
```

    
    The sky appears blue because of a phenomenon called Rayleigh scattering, which occurs when sunlight enters Earth's atmosphere. The Sun's light is made up of different colors, including red, orange, yellow, green, and violet. When these colors enter the atmosphere, they encounter tiny molecules of gases such as nitrogen and oxygen. These molecules scatter the light in all directions, but they scatter shorter (blue) wavelengths more than longer (red) wavelengths.
    
    As a result of this scattering, the blue light is dispersed throughout the atmosphere, giving the sky its blue appearance. The blue color we see is actually a combination of light that has been scattered in all directions by the tiny molecules of gases in the atmosphere. The amount of scattering that occurs depends on the wavelength of the light and the density of the gas molecules in the atmosphere, which is why the sky can appear different shades of blue depending on the time of day and atmospheric conditions.
    
    In addition to Rayleigh scattering, the sky can also appear blue due to the way that light interacts with the Earth's surface. When sunlight hits the ground, it can be reflected back into the atmosphere, adding to the overall blue color of the sky. This is why the sky often appears bluer in areas with a lot of vegetation or water, as these surfaces can reflect more light back into the atmosphere.
    
    Overall, the blue color of the sky is a result of the interaction between sunlight and the Earth's atmosphere, and it can vary depending on a variety of factors.

- `stream=True` 옵션을 사용하면 답변이 스트리밍 방식으로 출력된다.
- 스트리밍은 답변 생성이 완료되고 출력 되는것이 아닌 토큰이 생성될 떄 마다 출력해 주는 것을 의미한다.

### 3. 끝맺음
- Ollama와 python 라이브러리에 대해 간단하게 정리해보았다.
- 생성형 AI가 24년에 핫한 키워드인 만큼 점점 사용자에게 더 쉽게 사용할 수 있도록 발전되고 있다.
- 기본 성능은 ChatGPT에 비해 좋은 편은 아니나, RAG, Fine-tuning을 적용하여 사용자에게 커스터마이징된 생성형 AI를 사용한다면 굉장히 좋은 대안이 될 것으로 보인다.