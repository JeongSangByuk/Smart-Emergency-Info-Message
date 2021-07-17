from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re
import matplotlib as mpl
from keras_preprocessing.text import Tokenizer
from konlpy.tag import Mecab
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import pandas as pd

mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")

category = ['(코로나)안전수칙','(코로나)발생,방역','(코로나)동선','경제,금융','재난,날씨']
weather_important_keyword = ['지진','여진','태풍','한파','폭설','날씨','규모','산불','미세먼지','경보','건조','주의보','비상저감조치','산불','화재','기상']
economy_important_keyword=['금융', '생계', '소득', '지원금', '피싱', '경찰', '사칭', '통장', '계좌', '범죄']
number_category = len(category)

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

# 이외 데이터를 정제화 해주는 메소드
def refine_data(str):

    result = ""

    tmp_str = str[1:str.find(']')]
    important_location = ['환경부', '금융위원회', '행정안전부', '기상청']

    if tmp_str not in important_location:
        # 맨앞 [~~] 부분 제거
        str = str[str.find(']') + 1:]

    str = replace_keyword(str)
    temp_pos = mecab.pos(str)

    for j in range(len(temp_pos)):
        if (temp_pos[j][1] == 'NNG' or temp_pos[j][1] == 'NNP' or temp_pos[j][1] == 'VA' or temp_pos[j][1] == 'VV' \
                or temp_pos[j][1] == 'VV+EF' or temp_pos[j][1] == 'VA+ETM' or temp_pos[j][1] == 'XR'):
            result = result + " " + (temp_pos[j][0])
    return result

# tf/idf 평균,max 계산
def calcurate_average(data,str,cate):
    total = 0
    max_similarity = 0
    return_value = []
    keyword_number = 0

    for i in range(len(data)):
        doc_list = [data[i][0], str]
        tfidf_vectorizer = TfidfVectorizer(min_df=1)
        tfidf_matrix = tfidf_vectorizer.fit_transform(doc_list)
        doc_similarities = (tfidf_matrix * tfidf_matrix.T)

        #max 유사도 값 구하기
        if max_similarity<doc_similarities.toarray()[0][1]:
            max_similarity = doc_similarities.toarray()[0][1]

        total = total + doc_similarities.toarray()[0][1]

    ave = total / len(data)

    if cate == 'economy':
        for i in range(len(economy_important_keyword)):
            keyword_number = keyword_number + str.count(economy_important_keyword[i])

    elif cate == 'weather':
        for i in range(len(weather_important_keyword)):
            keyword_number = keyword_number + str.count(weather_important_keyword[i])

    else:
        keyword_number = 0

    max_similarity = max_similarity + (keyword_number*0.2)

    return_value.append(ave*100)
    return_value.append(max_similarity*100)

    return return_value

def calcurate_percentage(arr):
    total_value = sum(arr)

    for i in range(len(arr)):
        arr[i] = (arr[i]/total_value)*100

    return arr


# 최종 점수를 계산해주는 메소드
def calcurate_point(value):

    ave_similarity_array = [i[0] for i in value]
    max_similarity_array = [i[1] for i in value]
    result_array = []

    # 금융,경제 or 날씨, 재난일 경우  카테고리 0,1,2 의 포인트는 0, 4,5의 포인트 max_similarity 값으로 간다.
    if max_similarity_array.index(max(max_similarity_array)) == 3 or max_similarity_array.index(max(max_similarity_array)) == 4 :
        for i in range(3):
            result_array.append(0)
        temp_array = calcurate_percentage(max_similarity_array[3:5])
        for i in range(2):
            result_array.append(temp_array[i])

        return result_array

    # 코로나 관련 카테고리로 분류 일경우 우선 카테고리 3,4 의 포인트는 0으로 간다
    # 최대 유사도와 평균 유사도가 가르키는 카테고리가 같은 경우, (평균 유사도의 값 100% + 최대 유사도의 값 30%) 의 백분율을 이용하여 점수를 낸다.
    temp_arr = []

    if max_similarity_array.index(max(max_similarity_array)) == ave_similarity_array.index(max(ave_similarity_array)):

        for i in range(3):
            temp_arr.append(((max_similarity_array[i]*3)/10) + ave_similarity_array[i])

    # 최대 유사도와 평균 유사도가 가르키는 카테고리가 다를 경우, (평균 유사도의 값 100% + 최대 유사도의 값 100%) 의 백분율을 이용하여 점수를 낸다.
    else :
        for i in range(3):
            temp_arr.append(max_similarity_array[i] + ave_similarity_array[i])

    temp_2_arr = calcurate_percentage(temp_arr)

    for i in range(3):
        result_array.append(temp_2_arr[i])

    for i in range(2):
        result_array.append(0)

    return result_array

######################################## tfidf model

######################################### 단일 str 퍼센테이지로 출력

str = "[춘천시청] 코로나19 39번 확진자 관련 모든 관내 이동동선 접촉자 파악이 완료되어 검사진행중이며 이동경로는 방역소독을 완료하였습니다. /세부내용 홈페이지 참조"
str = refine_data(str)

value = []

value.append(calcurate_average(covid19_safe_rule_data, str, 'corona'))
value.append(calcurate_average(covid19_info_data, str, 'corona'))
value.append(calcurate_average(covid19_line_data, str, 'corona'))
value.append(calcurate_average(economy_data, str, 'economy'))
value.append(calcurate_average(weather_data, str, 'weather'))

print(str)

final_point = calcurate_point(value)
print('(코로나)안전수칙 : ', "%0.4f" % (final_point[0]), ' 발생,방역 : ', "%0.4f" % (final_point[1]), ' 동선 : ',
      "%0.4f" % (final_point[2]),
      '  경제,금융 : ', "%0.4f" % (final_point[3]), '  재난,날씨 : ', "%0.4f" % (final_point[4]), '\n')


########################테스트 데이터####################

##########################################  테스트 데이터 평가

# test_news= pd.read_csv('Data/disaster_msg_test.csv')
# msg_data = test_news['msg']
# msg_category = test_news['category']
#
# correct_num = 0
#
# for i in range(0,len(test_news)):
#     str = refine_data(msg_data[i])
#     total = 0
#     value = []
#
#     value.append(calcurate_average(covid19_safe_rule_data, str, 'corona'))
#     value.append(calcurate_average(covid19_info_data, str, 'corona'))
#     value.append(calcurate_average(covid19_line_data, str, 'corona'))
#     value.append(calcurate_average(economy_data, str, 'economy'))
#     value.append(calcurate_average(weather_data, str, 'weather'))
#
#     print(msg_data[i])
#
#     # print('(코로나)안전수칙 : ', "%0.4f"%(value[0][0]), ' 발생,방역 : ', "%0.4f"%(value[1][0]) , ' 동선 : ', "%0.4f"%(value[2][0]),
#     #       '  경제,금융 : ', "%0.4f"%(value[3][0]),'  재난,날씨 : ', "%0.4f"%(value[4][0]),'\n')
#     #
#     # print('(코로나)안전수칙 : ', "%0.4f"%(value[0][1]), ' 발생,방역 : ', "%0.4f"%(value[1][1]) , ' 동선 : ', "%0.4f"%(value[2][1]),
#     #       '  경제,금융 : ', "%0.4f"%(value[3][1]),'  재난,날씨 : ', "%0.4f"%(value[4][1]),'\n')
#
#     final_point = calcurate_point(value)
#     print('(코로나)안전수칙 : ', "%0.4f"%(final_point[0]), ' 발생,방역 : ', "%0.4f"%(final_point[1]) , ' 동선 : ', "%0.4f"%(final_point[2]),
#           '  경제,금융 : ', "%0.4f"%(final_point[3]),'  재난,날씨 : ', "%0.4f"%(final_point[4]),'\n')
#
#     if category[final_point.index(max(final_point))] == msg_category[i]:
#         correct_num+=1
#
# print( '정답률 : ', correct_num*100/len(test_news) , '%')





