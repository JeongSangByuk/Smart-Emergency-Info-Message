import pickle
import numpy as np
import tensorflow as tf
import os
import codecs
import pandas as pd
import konlpy
from konlpy.tag import Okt, Mecab
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Dense, LSTM,GlobalMaxPool1D
from tensorflow.keras.models import Sequential, save_model, load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Dropout, MaxPooling1D, Conv1D, Flatten, LeakyReLU, Bidirectional, \
    SpatialDropout1D
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import re
import json
from gensim.models import word2vec, KeyedVectors

############# 과정 ##############
# generator_token 을 이용해서 data token 생성
# 생성된 token을 이용해서 word2Vec model 생성
# 토크나이징 된 토큰과 워드 투백을 이용해서 learning_model
# 생성된 model을 이용해서 test_model
################################

okt = Okt()
mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")

data = pd.read_csv('../Data/data_msg_classified_csv.csv')

X = data['msg']
Y = data['category']

category = ['(코로나)발생,방역','(코로나)동선','안전수칙']
number_category = len(category)
max = 200
output_dim = 128

#라벨 인코딩 //encoder.classes_로 확인가능!
# transform 함수는 카테고리를 정수 인덱스 리스트로 변환
encoder = LabelEncoder()
encoder.fit(category)
Y = encoder.transform(Y)

#정수 인코딩 된 결과로부터 원-핫 인코딩을 수행하는 to_categorical 메소드
YoneHot = to_categorical(Y)

token = Tokenizer()
#pickle을 활용한 token 호출
with open('../Data/token_new_msg.pickle', 'rb') as handle:
    token = pickle.load(handle)

Xtoken = token.texts_to_sequences(X)
Xpad = pad_sequences(Xtoken,max)

#Embedding의 첫번째 인자값, 텍스트 데이터의 전체 단어 집합의 크기+1
wordsize = len(token.word_index) + 1

# ###################word2Vec 사용#############
# model_word2vec = word2vec.Word2Vec.load('Data/word2vec.model')
#
# def get_vector(word):
#     if word in model_word2vec:
#         return model_word2vec[word]
#     else:
#         return None
#
# embedding_vocab = np.zeros((wordsize,output_dim))
#
# for word,index in token.word_index.items():
#     # 단어(key) 해당되는 임베딩 벡터의 input_dim 의 값(value)를 임시 변수에 저장
#     temp = get_vector(word)
#     # 만약 none이 아니라면 벡터의 값을 리턴받은 것이므로 저장
#     if temp is not None:
#         embedding_vocab[index] = temp
# ##############################################

#train_test_split 함수는 전체 데이터셋 배열을 받아서 랜덤하게 훈련/테스트 데이터 셋으로 분리해주는 함수
#train_test_split 함수는 return 이 Xtrain,Xval,Ytrain,Yval 이다!

Xtrain,Xval,Ytrain,Yval = train_test_split(Xpad,YoneHot, test_size=0.2)

#################### 모델 설정 1
model = Sequential()
# 첫번째 인자 : 텍스트 데이터의 전체 단어 집합의 크기, 두번째 인자 : 임베딩 되고 난 후의 단어의 차원, 세번째 인자 : 입력 시퀀스의 길이
# Embedding() 단어를 밀집 벡터로 만드는 역할->임베딩 층을 만든다.
# word2Vec 사용시 임베딩
# model.add(Embedding(input_dim=wordsize,output_dim=output_dim,input_length=max,weights=[embedding_vocab],trainable=False))
# #그냥 임베딩
model.add(Embedding(input_dim=wordsize, output_dim=output_dim, input_length=max))
model.add(LSTM(60,return_sequences=True))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.2))
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(number_category, activation='softmax'))
################################################################

#################### 모델 설정 2
# model = Sequential()
# model.add(Embedding(wordsize,output_dim=output_dim,input_length=max,mask_zero=True))
# model.add(LSTM(64,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
# model.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
# model.add(Dense(number_category, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

hist = model.fit(Xtrain,Ytrain,batch_size=128,epochs=20,validation_data=(Xval,Yval),validation_split=0.2)

print("정확도 : %.4f" % (model.evaluate(Xtrain, Ytrain)[1]))

model.save('Data/model_msg.h5')

# ##샘플
# str = "2020.11.2.~2020.11.7.동안 11시부터 13시까지 인제하늘내린센터 헬스장 이용객분은 인제보건소에서 코로나19 검사 받으시기 바랍니다"
# sentence_new = token.texts_to_sequences([str])
# sent_pad_new = pad_sequences(sentence_new, max)
#
# arr = np.array(model.predict(sent_pad_new))
# result_print = ""
# for number_category in range(number_category):
#     result_print =result_print + category[number_category] + " : " + "%0.2f" %(arr[0][number_category]*100) + " , "
#
# print(result_print)

###############모델 테스트

# 중요 키워드 추출해주는 메소드
def replace_keyword(str):

    # 많이 쓰이는 방문 일자, 방문 시간에 대한 표현을 text로 반환 -> 동선에 대한 자료를 판단할 수 있게끔.
    # 1. dd:dd   2. dd시 dd분  3.  dd시   4. dd분
    # 1. dd.dd   2. dd월 dd일  3.  dd/dd  4. dd일
    # (월)~~(일), 월요일~일요일

    str = re.sub('번길', "", str)
    str = re.sub('\d{1,4}(번|명)'," 확진자누적번호 ",str)

    # 3자리 이상의 숫자 전부 삭제(전화번호, 주소 등 필요없는 정보삭제)
    str = re.sub('\d{3,}'," ",str)
    time = re.sub('\d{1,2}(:|시)(\d{1,2}(분*)){0,1}'," 방문시간 ",str)
    day = re.sub('\d{1,2}(\.|/|월)(\d{1,2}(일*)){0,1}'," 방문일자 ",time)
    week = re.sub('\(월\)|\(화\)|\(수\)|\(목\)|\(금\)|\(토\)|\(일\)|월요일|화요일|수요일|목요일|금요일|토요일|일요일'," 방문요일 ",day)
    return week

########################################### 퍼센테이지로 출력

str = "[익산시청]익산20번 이동동선 안내[1].10/27(화)11:00~14:00 식당주방근무,14:00~19:30시골집머뭄,20:00~20:20oo마트(마스크착용)"
str = str[str.find(']') + 1:]

result = []

# 맨앞 [~~] 부분 제거
str = str[str.find(']') + 1:]
str = replace_keyword(str)
temp_pos = mecab.pos(str)

for j in range(len(temp_pos)):
    if (temp_pos[j][1] == 'NNG' or temp_pos[j][1] == 'NNP' or temp_pos[j][1] == 'VA' or temp_pos[j][1] == 'VV' \
        or temp_pos[j][1] == 'VV+EF' or temp_pos[j][1] == 'VA+ETM' or temp_pos[j][1] == 'XR') and temp_pos[j][0]:
        result.append(temp_pos[j][0])

str = result

sentence = token.texts_to_sequences(str)
sent_pad = pad_sequences(sentence, max)

arr = np.array(model.predict(sent_pad))
result_print = ""
for number_category in range(number_category):
    result_print =result_print + category[number_category] + " : " + "%0.2f" %(arr[0][number_category]*100) + " , "

print(result_print)

