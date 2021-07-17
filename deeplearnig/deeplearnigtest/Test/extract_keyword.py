# 필요한 패키지와 라이브러리를 가져옴
import pickle
import re
import matplotlib as mpl
from keras_preprocessing.text import Tokenizer
from konlpy.tag import Mecab
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")

#np.load 가 보안의 문제로 막혀있을 경우, allow_pickle=True 로 설정
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
total_data_msg = np.load('Data/CategoryData/total_data_array.npy')
covid19_line_data = np.load('Data/CategoryData/covid19_line_data_array.npy')
covid19_safe_rule_data = np.load('Data/CategoryData/covid19_safe_rule_data_array.npy')
covid19_info_data = np.load('Data/CategoryData/covid19_info_data_array.npy')
economy_data = np.load('Data/CategoryData/economy_data_array.npy')
weather_data = np.load('Data/CategoryData/weather_data_data_array.npy')
np.load = np_load_old

# min_df = 2 -> 문장에서 1번 미만으로 나타나는 단어 무시.
vectorize_1 = CountVectorizer(min_df=15)
vectorize_2 = CountVectorizer(min_df=15)
vectorize_3 = CountVectorizer(min_df=15)
vectorize_4 = CountVectorizer(min_df=5)
vectorize_5 = CountVectorizer(min_df=5)

def refine_array(data):
    arr = []
    for i in range(len(data)):
        arr.append(data[i][0])
    return arr

X_caet1 = vectorize_1.fit_transform(refine_array(covid19_line_data))
X_caet2 = vectorize_2.fit_transform(refine_array(covid19_safe_rule_data))
X_cate3 = vectorize_3.fit_transform(refine_array(covid19_info_data))
X_caet4 = vectorize_4.fit_transform(refine_array(economy_data))
X_cate5 = vectorize_5.fit_transform(refine_array(weather_data))

# 문장에서 뽑아낸 feature 들의 배열
features_1 = vectorize_1.get_feature_names()
features_2 = vectorize_2.get_feature_names()
features_3 = vectorize_3.get_feature_names()
features_4 = vectorize_4.get_feature_names()
features_5 = vectorize_5.get_feature_names()


print(features_1)
print(features_2)
print(features_3)
print(features_4)
print(features_5)

