import pickle
import numpy as np
import tensorflow as tf
import os
import codecs
import pandas as pd
import konlpy
from konlpy.tag import Okt
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Dense, LSTM, GlobalMaxPool1D
from tensorflow.keras.models import Sequential, save_model, load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Dropout, MaxPooling1D, Conv1D, Flatten
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import re
import json
from gensim.models import word2vec
from konlpy.tag import Mecab

token = Tokenizer()
mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")
hangul = re.compile('[^0-9a-zA-Zㄱ-힗]+')

data = pd.read_csv('Data/data_msg_classified_csv.csv')
stopwords = pd.read_csv('Data/stopword_msg.csv')

msg_data = data['msg']
category_data = data['category']
stoplist = list(stopwords['stopword'])


# 중요 키워드 추출해주는 메소드
def replace_keyword(str):
    # 많이 쓰이는 방문 일자, 방문 시간에 대한 표현을 text로 반환 -> 동선에 대한 자료를 판단할 수 있게끔.
    # 1. dd:dd   2. dd시 dd분  3.  dd시   4. dd분
    # 1. dd.dd   2. dd월 dd일  3.  dd/dd  4. dd일
    # (월)~~(일), 월요일~일요일

    str = re.sub('번길', "", str)
    str = re.sub('\d{1,4}(번|명)', " 확진자누적번호 ", str)

    # 3자리 이상의 숫자 전부 삭제(전화번호, 주소 등 필요없는 정보삭제)
    str = re.sub('\d{3,}', " ", str)
    time = re.sub('\d{1,2}(:|시)(\d{1,2}(분*)){0,1}', " 방문시간 ", str)
    day = re.sub('\d{1,2}(\.|/|월)(\d{1,2}(일*)){0,1}', " 방문일자 ", time)
    week = re.sub('\(월\)|\(화\)|\(수\)|\(목\)|\(금\)|\(토\)|\(일\)|월요일|화요일|수요일|목요일|금요일|토요일|일요일', " 방문요일 ", day)
    return week


# 형태소분리
for i in range(len(msg_data)):

    result = []

    tmp_str = msg_data[i][1:msg_data[i].find(']')]
    important_location = ['환경부', '금융위원회', '행정안전부']

    if tmp_str not in important_location:
        # 맨앞 [~~] 부분 제거
        msg_data[i] = msg_data[i][msg_data[i].find(']') + 1:]

    msg_data[i] = replace_keyword(msg_data[i])
    temp_pos = mecab.pos(msg_data[i])

    for j in range(len(temp_pos)):

        if (temp_pos[j][1] == 'NNG' or temp_pos[j][1] == 'NNP' or temp_pos[j][1] == 'VA' or temp_pos[j][1] == 'VV' \
            or temp_pos[j][1] == 'VV+EF' or temp_pos[j][1] == 'VA+ETM' or temp_pos[j][1] == 'XR') and temp_pos[j][0] not in stoplist:
            result.append(temp_pos[j][0])

    msg_data[i] = result

    # str_result = ""
    # for j in range(len(result)):
    #     str_result = str_result + " " + result[j]
    #
    # msg_data[i] = str_result
    # print(category_data[i], "     :      ", msg_data[i])

# fit_on_texts 는 단어 인덱스를 구축
# texts_to_sequences 은 정수 인덱스를 리스트로 변환
# pad_sequences 길이가 같지 않고 적거나 많을 때 일정한 길이로 맞춰 줄 때 사용
print(msg_data)
token.fit_on_texts(msg_data)

# 분석한 token을 저장해서 모델을 사용할때 같이 불러줘야한다.그렇지 않으면 모델로 사용할 때 예측을 하고픈 문장으로
# 토크나이징이 되기 때문에 학습의 결과가 아무 의미가 없다.
# 데이타 토큰 생성
# with open('Data/token_new_msg.pickle','wb') as handle:
#     pickle.dump(token,handle,protocol=pickle.HIGHEST_PROTOCOL)

# data_array.npy 생성
np.save('Data/data_msg_array.npy',msg_data)
# np.save('Data/data_category_array.npy',category_data)

# a = np.array(msg_data)
# b = np.array(category_data)
#
# # 두배열 이중배열로 합치기
# arr_result = np.c_[a, b]
#
# covid19_line_data = []
# covid19_safe_rule_data = []
# covid19_info_data = []
# economy_data = []
# weather_data = []
#
# for i in range(len(arr_result)):
#     if arr_result[i][1] == '(코로나)동선':
#         covid19_line_data.append(arr_result[i])
#     elif arr_result[i][1] == '(코로나)안전수칙':
#         covid19_safe_rule_data.append(arr_result[i])
#     elif arr_result[i][1] == '(코로나)발생,방역':
#         covid19_info_data.append(arr_result[i])
#     elif arr_result[i][1] == '경제,금융':
#         economy_data.append(arr_result[i])
#     elif arr_result[i][1] == '재난,날씨':
#         weather_data.append(arr_result[i])
#
# # 전체 배열 저장, 카테고리 별로 저장
# np.save('Data/CategoryData/total_data_array.npy', arr_result)
# np.save('Data/CategoryData/covid19_line_data_array.npy', covid19_line_data)
# np.save('Data/CategoryData/covid19_safe_rule_data_array.npy', covid19_safe_rule_data)
# np.save('Data/CategoryData/covid19_info_data_array.npy', covid19_info_data)
# np.save('Data/CategoryData/economy_data_array.npy', economy_data)
# np.save('Data/CategoryData/weather_data_data_array.npy', weather_data)
