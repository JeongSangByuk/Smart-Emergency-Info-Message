import re

import numpy as np
import tensorflow as tf
import json
import codecs
import pandas as pd
import konlpy
from konlpy.tag import Okt
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
import pickle

okt = Okt()

data = pd.read_csv('c:/users/BYUK/desktop/100sett.csv')
stopwords = pd.read_csv('c:/users/BYUK/desktop/stop_word.csv',index_col=0)
hangul = re.compile('[^0-9a-zA-Zㄱ-힗]+')

X = data['msg']

stoplist = list(stopwords['stopword'])

category = ['(코로나)발생,방역','(코로나)동선','안전수칙','지진','산불','경제','범죄']
number_category = len(category)
max = 200

#불용어 제거를 위한 형태소 분리
for i in range(len(X)):
    X[i] = okt.morphs(X[i])

# 불용어가 아닌 단어만 result에 넣어서
for i in range(len(X)):
    result = []
    for j in range(len(X[i])):
        if X[i][j] not in stoplist:
            result.append(X[i][j])
    X[i] = result

# 불용어 제거 후 다시 문장 만들기
for i in range(len(X)):
    result = ""
    for j in range(len(X[i])):
        result = result + " " + X[i][j]
    result = hangul.sub(" ", result)
    X[i] = result

#fit_on_texts 는 단어 인덱스를 구축
#texts_to_sequences 은 정수 인덱스를 리스트로 변환
#pad_sequences 길이가 같지 않고 적거나 많을 때 일정한 길이로 맞춰 줄 때 사용
token = Tokenizer()
token.fit_on_texts(X)

#분석한 token을 저장해서 모델을 사용할때 같이 불러줘야한다.그렇지 않으면 모델로 사용할 때 예측을 하고픈 문장으로
#토크나이징이 되기 때문에 학습의 결과가 아무 의미가 없다.
# word_index = token.word_index
# json = json.dumps(word_index)
# f3 = open("word.json", "w")
# f3.write(json)
# f3.close()

# data_array.npy 생성
# np.save('data_array.npy',X)

#데이타 토큰 생성
with open('Data/','wb') as handle:
    pickle.dump(token,handle,protocol=pickle.HIGHEST_PROTOCOL)

