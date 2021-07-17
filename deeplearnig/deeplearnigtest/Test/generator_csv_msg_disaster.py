import json
import os
from urllib.parse import urlencode, quote_plus, unquote
from urllib.request import Request, urlopen
import csv
import numpy as np
import tensorflow as tf
import requests
from bs4 import BeautifulSoup

################### API로 데이터 호출
key = 'HyrybDIOQCqNNI%2BmdazoNhjLwQ%2BdnR8G04jK%2Bmu3aeouMtRbraG5U%2FwdpBlJwaI9cocl27x4VXd%2BQe62tJ8KNA%3D%3D'
decode_key = unquote(key)

url = 'http://apis.data.go.kr/1741000/DisasterMsg2/getDisasterMsgList'
queryParams = '?' + urlencode({ quote_plus('ServiceKey') : decode_key, quote_plus('pageNo') : '1', quote_plus('numOfRows') : '10', quote_plus('type') : 'json', quote_plus('flag') : 'Y' })

request = Request(url + queryParams)
request.get_method = lambda: 'GET'
response_body = urlopen(request).read()
print(response_body)
# #json 파일로 저장
# with open('disasterMsg.josn','w',encoding = 'utf-8') as make_json:
#     json.dump(json_str,make_json, indent = "/t",ensure_ascii = False)
# #
# ###########################################
# with open('disasterMsg.josn','r',encoding = 'utf-8') as f:
#     json_data = json.load(f)
#
# # dict으로 변경
# dict = json.loads(json_data)
#
# dict_ = dict['DisasterMsg'][1]['row']
#
# location = []
# msg = []
#
# for k in dict_:
#     location.append(k['location_name'].replace("\n"," "))
#     msg.append(k['msg'].replace("\n",""))
#
# arr = np.c_[location,msg]
#
# with open('disaster_msg.csv','w',newline='') as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerow(["location","msg"])
#     for i in arr:
#         writer.writerow(i)


