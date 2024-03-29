# ✉️ Smart-Emergency-Info-Message
<img src = "https://img.shields.io/badge/ProjectType-TeamProject-orange?style=flat-square">  <img src = "https://img.shields.io/badge/Tools-AndroidStudio-brightgreen?style=flat-square&logo=AndroidStudio"> <img src = "https://img.shields.io/badge/Tools-VScode-brightgreen?style=flat-square&logo=VisualStudioCode"> <img src = "https://img.shields.io/badge/Tools-Pycharm-brightgreen?style=flat-square&logo=Pycharm"> <img src = "https://img.shields.io/badge/Tools-PaasTa-brightgreen?style=flat-square"> <br> <img src = "https://img.shields.io/badge/Language-Java-critical?style=flat-square&logo=Java"> <img src = "https://img.shields.io/badge/Language-Javascript-critical?style=flat-square&logo=Javascript"> <img src = "https://img.shields.io/badge/Language-Python-critical?style=flat-square&logo=Python">
> ‘Smart 재난문자’는 기존 재난문자의 불편함을 해소하고 편의성을 개선하여, <br>재난문자의 본래 목적을 효과적으로 달성할 수 있는 서비스입니다.
<p align="center"> <img src = "https://user-images.githubusercontent.com/64072741/126045059-616dcde1-4210-462b-8ffc-7c0a98314a0e.png"> </p>

##  💡  Background
1. **재난문자 수신양의 급증** : 행정안전부에 따르면 2020년 1~9월 말 송출된 재난문자는 3만 4679건, 일일 평균 126건이라고 합니다. 이는 지난해보다 약 18배나 폭증한 양입니다.
2. **재난문자 발송 방식의 단점 - 위치적으로 연관 없는 재난 문자 수신** : 자신과 위치적으로 연관 없는 재난문자를 수신 할 수 있으며, 중복된 문자를 받을 수 있습니다.
3. **재난문자의 카테고리 유용성 판단에 따른 피로도 증가** : 어떤 카테고리의 재난문자의 정보가 더 유용한지, 시급한지는 사용자가 판단해야합니다. 그리고 그 유용성을 판단하는 순간은 분명 사용자에게 피로와 부담으로 다가 올 것입니다. 

#### 이러한 세 가지 배경으로 인해 초래된 가장 큰 문제는 바로 국민들이 재난 문자에 대해 무감각해지는 것입니다.<br>많은 사람이 재난문자 알림 꺼버리고 예사롭지 않게 읽고 넘기는 분위기는 새로운 안전 사각 지대를 발생 시켜 더 큰 문제를 가져올 것입니다.<br><br>

##  📝  Features
### 1. 위치 기반 사용자 맞춤 재난문자 
기존의 송출 기지국 반경 15km까지 발송되는 시스템은 사용자 개인과 연관성이 떨어지는 문자가 올 가능성이 높습니다. ‘Smart 재난문자’는 수신 지역 추가 또는 마이데이터 분석을 통해 재난문자발송 지역을 특정, 더 정확한 재난문자를 수신할 수 있습니다.
### 2. 카테고리 선호도 기반 재난문자
재난문자 발생 즉시, 재난 문자 분석 AI를 통해 5가지 유형으로 분류되어 사용자에게 송출됩니다. 사용자는 푸시알림을 통해 이를 확인하여 수신 즉시 자신에게 유용성을 판단 할 수 있습니다.
### 3. 사용자 맞춤 알림 서비스
사용자는 분류된 문자 유형에 따라 레벨을 조절할 수 있습니다. 이 레벨에 따라 정부에서 문자 발송 시 푸시알림 여부, 소리/진동 등을 상세히 조절할 수 있어 사용자에게 더 필요한 재난문자를 맞춤으로 제공합니다.
### 4. 한눈에 보는 코로나 정보
재난문자로 지역감염을 알게 되어도 지역홈페이지를 따로 검색해야했던 불편함을 해소하기 위해 재난문자 터치 시 바로 지자체·정부 사이트로 이동 기능, 일일 코로나 현황판 등 사용자 편의성을 높이는 기능을 완성도 있게 제공합니다.<br><br>

##  📚  Stack & Library
+ Android/Java
+ Paas-Ta/MySQL
+ Aws
+ Express
+ Tensorflow : NLP 머신 러닝 모델 구현을 위해 사용.
+ Scikt Learn : NLP 머신 러닝 모델 구현을 위해 사용.
+ Firebase : 실시간 알림 서비스를 위해 사용.
+ Room : 안드로이드 로컬 DB 사용.<br><br>

##  🖥️  Preview
### 1. 위치 기반 사용자 맞춤 재난문자.
<p align="center"> <img src = "https://user-images.githubusercontent.com/64072741/126047408-32e4b4af-56b8-49bb-98ce-5e1895712708.png"> </p>

### 2. 카테고리 선호도 기반 재난문자 알림.
<p align="center"> <img src = "https://user-images.githubusercontent.com/64072741/126047436-9aa2613c-5612-470c-97f4-37d54c685149.png"> </p>

### 3. 사용자 맞춤 알림 서비스.
<p align="center"> <img src = "https://user-images.githubusercontent.com/64072741/126047303-fda72996-2e06-40c8-a8c6-a63b2dd36d53.png"> </p>

### 4. 한눈에 보는 코로나 정보.
<p align="center"> <img src = "https://user-images.githubusercontent.com/64072741/126047412-99bac4b3-ba8b-4190-bbf7-3cae33df8c54.png"> </p><br>

##  🛠️  Architecture

### 개발 환경 구성도

<p align="center"> <img src = "https://user-images.githubusercontent.com/64072741/126046207-bd365d29-d17a-4ac7-aeea-49fee9912a12.png"> </p>

1. 안드로이드 네이티브 어플리케이션을 개발하였습니다.
3. Paas-ta 의 node.js 서버를 통해 어플리케이션에서 데이터를 주고받습니다.
4. 데이터들은 Paas-ta의 node.js 서버에 바인딩 된 MySQL 서비스를 통해 저장되고 사용자에게 전송됩니다.
5. Paas-ta 어플리케이션과 서비스를 CLOUD FOUNDERY를 통해 MySQL에 접속하고 소스코드를 푸시하는 등 관리·개발 하였습니다.
6. 재난문자를 가져오고 분석하여 분류하는 머신러닝기술 활용을 위해 AWS서버에 Tensor Flow와 Scikit Learn을 이용한 Machine Learning Model을 올렸습니다.
7. 위 과정을 통해 정제된 재난문자는 MySQL DB에 저장됨과 동시에  Firebase FCM 기술을 활용해 알맞은 사용자에게 푸시 알림을 전송하였습니다.<br><br>

### 서비스 구성도
<p align="center"> <img src = "https://user-images.githubusercontent.com/64072741/126046588-98e47cb9-a36b-40d2-9ee7-deae588e89a5.png"> </p>

1. 재난문자의 카테고리를 ‘(코로나) 동선, (코로나) 발생/방역, (코로나) 안전수칙, 재난/날씨, 경제, 금융’ 로 5종류의 카테고리로 확립하였습니다.
2. 2020년 12월 1일부터 과거까지 재난문자 데이터 대략 1,000개 정도를 훈련 데이터 셋으로 수집한 후 일일이 분류하여 훈련 데이터셋을 확보했습니다.
3. 재난문자 카테고리 분류를 하기위해 Skcit Learn의 TF/IDF 기법을 이용한 Pretrained Model을 도입했으며,<br> LSTM 모델에 Pretrained Model을 Embedding하고 feature를 연속해서 통합하는 방식을 이용했습니다.
4. Level을 도출하기 위해 가중치 부여 알고리즘을 채택했습니다. 사용자가 높게 설정한 카테고리와 사용자 위치 연관성이 높은 재난문자에 가중치를 부여합니다. 
5. 사용자가 설정을 수정하는 즉시 관리되고 있는 모든 재난문자의 Level이 재할당되며, 사용자 UI도 이에 맞게 업데이트됩니다.<br><br>


## 🎓 I Learned
+ 처음으로 접한 NLP 머신 러닝 개발이었습니다. 처음 접하는 분야이기 때문에 개발하면서 많은 어려움이 있었음에도 개발한 모델의 성능이 꽤나 만족스럽게 나와서 매우 뿌듯했습니다. 머신 러닝 시스템의 프로세스를 경험해볼 수 있는 좋은 기회였습니다.
+ 재난문자가 관리되는 주요 뷰의 전반적인 UI/UX를 설계했습니다. 그 과정에서 슬라이딩업 패널을 도입했는데, 사용자 입장에서 세련되고 퀄리티있는 뷰를 제공하기 위해 많은 노력을 한 것 같습니다.
+ 각각의 재난문자가 사용자의 설정(위치, 카테고리 선택 유무, 카테고리 가중치)에 따라 동적으로 관리 및 처리되면서, 메인 액티비티의 재난문자 리사이클러뷰에 대한 아주 섬세한 처리가 필요했습니다. 재난문자와 관련된 모든 value들을 고려하면서 기능의 로직을 개발하기 위해 많은 시간을 사용했습니다. 
+ 또한 위의 과정에서 발생했던 많은 예외를 잡기 위해서 그리고 조금 더 효율적인 로직 플로우 짜기 위해서 많은 고민을 했습니다.
+ 완성도를 높이기 위해 많은 노력을 한 프로젝트여서 개인적으로 매우 뿌듯하고 자신있는 프로젝트입니다.<br><br>

## 🔍 More
<p align="center"> <a href="https://www.youtube.com/watch?v=tpk337-h3ZE"><img src="https://user-images.githubusercontent.com/64072741/126046383-4420ad00-a2f4-48da-a94c-6f15f75e5490.png"/></a> </p>



