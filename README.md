# ✉️ Smart-Emergency-Info-Message
<img src = "https://img.shields.io/badge/ProjectType-TeamProject-orange?style=flat-square">  <img src = "https://img.shields.io/badge/Tools-AndroidStudio-brightgreen?style=flat-square&logo=AndroidStudio"> <img src = "https://img.shields.io/badge/Tools-VScode-brightgreen?style=flat-square&logo=VisualStudioCode"> <img src = "https://img.shields.io/badge/Tools-Pycharm-brightgreen?style=flat-square&logo=Pycharm"> <img src = "https://img.shields.io/badge/Tools-PaasTa-brightgreen?style=flat-square"> <br> <img src = "https://img.shields.io/badge/Language-Java-critical?style=flat-square&logo=Java"> <img src = "https://img.shields.io/badge/Language-Javascript-critical?style=flat-square&logo=Javascript"> <img src = "https://img.shields.io/badge/Language-Python-critical?style=flat-square&logo=Python">
> ‘Smart 재난문자’는 기존 재난문자의 불편함을 해소하고 편의성을 개선하여, <br>재난문자의 본래 목적을 효과적으로 달성할 수 있는 서비스입니다.
<p align="center"> <img src = "https://user-images.githubusercontent.com/64072741/126045059-616dcde1-4210-462b-8ffc-7c0a98314a0e.png"> </p>

##  💡  Background
1. **재난문자 수신양의 급증** : 행정안전부에 따르면 2020년 1~9월 말 송출된 재난문자는 3만 4679건, 일일 평균 126건이라고 한다. 지난해보다 약 18배나 폭증한 양입니다.
2. **재난문자 발송 방식의 단점 - 위치적으로 연관 없는 재난 문자 수신** : 자신과 위치적으로 연관 없는 재난문자를 수신 할 수 있으며, 중복된 문자를 받을 수 있습니다.
3. **재난문자의 카테고리 유용성 판단에 따른 피로도 증가** : 어떤 카테고리의 재난문자의 정보가 더 유용한지, 시급한지는 사용자가 판단해야합니다. 그리고 그 유용성을 판단하는 순간은 분명 사용자에게 피로와 부담으로 다가 올 것입니다. 

#### 이러한 세 가지 배경으로 인해 초래된 가장 큰 문제는 바로 국민들이 재난 문자에 대해 무감각해지는 것입니다.<br>많은 사람이 재난문자 알림 꺼버리고 예사롭지 않게 읽고 넘기는 분위기는 새로운 안전 사각 지대를 발생 시켜 더 큰 문제를 가져올 것입니다.<br><br>

##  📝  Features
### 1. 위치 기반 사용자 맞춤 재난문자
기존의 송출 기지국 반경 15km까지 발송되는 시스템은 사용자 개인과 연관성이 떨어지는 문자가 올 가능성이 높습니다. ‘Smart 재난문자’는 수신 지역 추가 또는 마이데이터 분석을 통해 재난문자발송 지역을 특정, 더 정확한 재난문자를 수신할 수 있습니다.
### 2. 카테고리 선호도 기반 재난문자
재난문자 발생 즉시, 재난 문자 분석 AI를 통해 5가지 유형으로 분류되어 사용자에게 송출됩니다. 사용자는 푸시알림을 통해 이를 확인하여 수신 즉시 자신에게 유용성을 판단 할 수 있습니다.
### 3. 사용자 맞춤 알림 서비
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
<p align="center"> <img src = "https://user-images.githubusercontent.com/64072741/126046010-25df2891-269c-4ee7-ab9e-7ddfb7e05994.png"> </p>

### 2. 카테고리 선호도 기반 재난문자 알림.
<p align="center"> <img src = "https://user-images.githubusercontent.com/64072741/126045996-db388023-0e04-493c-83e5-2698c36f6a09.png"> </p>

### 3. 사용자 맞춤 알림 서비스.
<p align="center"> <img src = "https://user-images.githubusercontent.com/64072741/126046075-cab43414-d2f2-456c-8fde-058e6f96efc5.png"> </p>

### 4. 한눈에 보는 코로나 정보.
<p align="center"> <img src = "https://user-images.githubusercontent.com/64072741/126046079-340c8651-cdbf-496e-8de8-dae4565a299d.png"> </p><br>

##  🛠️  Architecture
<p align="center"> <img src = "https://user-images.githubusercontent.com/64072741/126046207-bd365d29-d17a-4ac7-aeea-49fee9912a12.png"> </p>

1. 안드로이드 네이티브 어플리케이션을 개발하였습니다.
3. Paas-ta 의 node.js 서버를 통해 어플리케이션에서 데이터를 주고받습니다.
4. 데이터들은 Paas-ta의 node.js 서버에 바인딩 된 MySQL 서비스를 통해 저장되고 사용자에게 전송됩니다.
5. Paas-ta 어플리케이션과 서비스를 CLOUD FOUNDERY를 통해 MySQL에 접속하고 소스코드를 푸시하는 등 관리·개발 하였습니다.
6. 재난문자를 가져오고 분석하여 분류하는 머신러닝기술 활용을 위해 AWS서버에 Tensor Flow와 Scikit Learn을 이용한 Machine Learning Model을 올렸습니다.
7. 위 과정을 통해 정제된 재난문자는 MySQL DB에 저장됨과 동시에  Firebase FCM 기술을 활용해 알맞은 사용자에게 푸시 알림을 전송하였습니다.<br>

## 🎓 I Learned

## 🔍 More
<a href="https://www.youtube.com/watch?v=tpk337-h3ZE"><img src="https://user-images.githubusercontent.com/64072741/126046383-4420ad00-a2f4-48da-a94c-6f15f75e5490.png"/></a>
