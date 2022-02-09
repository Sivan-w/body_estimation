import requests
import json
import cv2
import base64

import numpy as np


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


# 发送HTTP请求
org_im = cv2.imread('images.jpg') # 选择单张图片位置或批量图片文件夹路径
data = {'images':[cv2_to_base64(org_im)]}
headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/openpose_body_estimation" # 本机默认地址
r = requests.post(url=url, headers=headers, data=json.dumps(data))
canvas = base64_to_cv2(r.json()["results"]['data'])
cv2.imwrite('keypoint_body.png', canvas) #保存