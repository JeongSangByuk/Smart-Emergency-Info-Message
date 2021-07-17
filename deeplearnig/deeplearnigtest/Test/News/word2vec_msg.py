import numpy as np
from gensim.models import Word2Vec,KeyedVectors
import matplotlib.pyplot as plot
from sklearn.decomposition import PCA
from matplotlib import font_manager

#np.load 가 보안의 문제로 막혀있을 경우, allow_pickle=True 로 설정
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
data_news = np.load('../Data/data_msg_array.npy')
np.load = np_load_old



input_dim= 128
#word2Vec 모델
#window = 앞뒤 단어 5개라는 의미, min_count=3 출현빈도 3개 이하는 제외
model = Word2Vec(sentences=data_news, size=input_dim, window=5, min_count=1, sg=1)

mode_result = model.wv.most_similar("확진환자")
print(mode_result)

# model.save('Data/word2vec.model')

############################# W2V 시각화 ######################################
# matplotlib 한글 폰트를 지정하는부분 -> 폰트 지정안하면 깨짐ㅅㄱ
font_manager.get_fontconfig_fonts()
ont_location = 'C:/Windows/Fonts/batang.ttc'
font_name = font_manager.FontProperties(fname=ont_location,size=10).get_name()
plot.rc('font', family=font_name)
plot.rc('axes', unicode_minus=False)

word_vectors = model.wv
vocabs = word_vectors.vocab.keys()
word_vectors_list = [word_vectors[v] for v in vocabs]
pca = PCA(n_components=2)
xys = pca.fit_transform(word_vectors_list)
xs= xys[:,0]
ys = xys[:,1]

plot.figure(figsize=(10,8))
plot.scatter(xs,ys,marker='o')
for i,v in enumerate(vocabs):
    plot.annotate(v,xy=(xs[i],ys[i]))

plot.show()
############################# W2V 시각화 ######################################