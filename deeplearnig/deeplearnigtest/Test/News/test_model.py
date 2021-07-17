import re
import numpy as np
import tensorflow as tf
import pickle
import codecs
import pandas as pd
import konlpy
from konlpy.tag import Okt, Mecab
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.layers import Dropout, MaxPooling1D, Conv1D, Flatten
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, save_model, load_model
import json
from keras_preprocessing.text import tokenizer_from_json

okt = Okt()

#모델 불러오기
model = load_model('../Data/News/model_msg.h5')

category = ['(코로나)발생,방역','(코로나)동선','안전수칙']
number_category = len(category)

stopwords = pd.read_csv('../Data/stopword_msg.csv')
stoplist = list(stopwords['stopword'])

max = 200

token = Tokenizer()

#pickle을 활용한 token 호출
with open('../Data/token_new_msg.pickle', 'rb') as handle:
    token = pickle.load(handle)

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

str = "[성남시청] 10.25(일) 10:00~11:30, 26(월) 10:15~11:40 스파밸리(율동) 남자사우나 방문객 중 유증상자는 선별진료소에서 검사받으세요."
str = str[str.find(']') + 1:]

result = []

# 맨앞 [~~] 부분 제거
str = str[str.find(']') + 1:]
str = replace_keyword(str)
temp_pos = mecab.pos(str)

for j in range(len(temp_pos)):
    if (temp_pos[j][1] == 'NNG' or temp_pos[j][1] == 'NNP' or temp_pos[j][1] == 'VA' or temp_pos[j][1] == 'VV' \
        or temp_pos[j][1] == 'VV+EF' or temp_pos[j][1] == 'VA+ETM' or temp_pos[j][1] == 'XR') :
        result.append(temp_pos[j][0])

str = result
print(str)
sentence = token.texts_to_sequences(str)
sent_pad = pad_sequences(sentence, max)

arr = np.array(model.predict(sent_pad))
result_print = ""
for number_category in range(number_category):
    result_print =result_print + category[number_category] + " : " + "%0.2f" %(arr[0][number_category]*100) + " , "

print(result_print)

###########################################  테스트 데이터 평가

# test_news= pd.read_csv('c:/users/BYUK/desktop/today_news_titles.csv',index_col=0)
# T_X = test_news['title']
#
# test_news['predicted'] = 0
# test_news['OX'] = 'O'
# hangul = re.compile('[^0-9a-zA-Zㄱ-힗]+')
#
# onum = 0
# xnum = 0
#
# for i in range(len(T_X)):
#
#     sentence = test_news['title'][i]
#     sentence = hangul.sub(" ", sentence)
#     sentence = okt.morphs(sentence)
#     result = []
#     for j in range(len(sentence)):
#         if sentence[j] not in stopwords:
#             result.append(sentence[j])
#     sentence = result
#     result = ""
#     for k in range(len(sentence)):
#         result = result + " " + sentence[k]
#     sentence = result
#     sentence = token.texts_to_sequences([sentence])
#     sent_pad = pad_sequences(sentence, max)
#
#     test_news['predicted'][i] = category[np.argmax(model.predict(sent_pad))]
#
#     if test_news['category'][i] == test_news['predicted'][i]:
#         test_news['OX'][i] = 'O'
#         onum+=1
#     else:
#         test_news['OX'][i] = 'X'
#         xnum+=1
#
# for i in range(len(T_X)):
#     print(test_news['category'][i] + "    " + test_news['predicted'][i] + "    "  + test_news['OX'][i])
# print('accuracy is :',onum/(onum+xnum)*100)
