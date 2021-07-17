import pickle
import numpy as np
import tensorflow as tf
import os
import codecs
import pandas as pd
import konlpy
from konlpy.tag import Okt
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Dense, LSTM,GlobalMaxPool1D
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
from gensim.models import Word2Vec,KeyedVectors

#np.load 가 보안의 문제로 막혀있을 경우, allow_pickle=True 로 설정
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
data_news = np.load('Data/data_array.npy')
np.load = np_load_old

#word2Vec 모델
#window = 앞뒤 단어 5개라는 의미, min_count=3 출현빈도 3개 이하는 제외
model = Word2Vec(sentences=data_news, size=64, window=5, min_count=3, sg=1)

#mode_result = model.wv.most_similar("정치")

# model.save('word2vec.model')



