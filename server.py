from flask import Flask, make_response, render_template, request
from turbojpeg import TurboJPEG
import paddlehub as hub
import numpy as np
import base64
import requests
import os
import cv2

app = Flask(__name__)
model = hub.Module(name='openpose_body_estimation')


# def cv2_to_base64(image):
#     data = cv2.imencode('.jpg', image)[1]
#     return base64.b64encode(data.tostring()).decode('utf8')
#
#
# def base64_to_cv2(b64str):
#     data = base64.b64decode(b64str.encode('utf8'))
#     data = np.fromstring(data, np.uint8)
#     data = cv2.imdecode(data, cv2.IMREAD_COLOR)
#     return data


# def load_turbojpeg():
#     basePath = os.path.split(os.path.realpath(__file__))[0]
#     TurboJPEGPath = basePath + '\\turbojpeg.dll'
#     turbojpeg = TurboJPEG(TurboJPEGPath)
#     return turbojpeg
#
#
# image_focus = 14521.6
# turbojpeg = load_turbojpeg()


@app.route('/index')
def indexes():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def uploadImg():
    if request.method=='GET':
        return  render_template('index.html')
    elif request.method == 'POST':
        if 'myImg' in request.files:
            objFile = request.files.get('myImg')
            objFile.filename = "img.jpg"
            strFileName = objFile.filename
            strFilePath = "./static/imgs/"+strFileName
            objFile.save(strFilePath)
            # # 发送HTTP请求
            # org_im = cv2.imread('images.jpg')  # 选择单张图片位置或批量图片文件夹路径
            # data = {'images': [cv2_to_base64(org_im)]}
            # headers = {"Content-type": "application/json"}
            # url = "http://127.0.0.1:8866/predict/openpose_body_estimation"  # 本机默认地址
            # r = requests.post(url=url, headers=headers, data=json.dumps(data))
            # canvas = base64_to_cv2(r.json()["results"]['data'])
            # cv2.imwrite('../static/imgs/output.png', canvas)  # 保存

            result = model.predict('D:\\project\\pythonPrj\\bodyEstimation\\static\\imgs\\' + strFileName, 'D:\\project\\pythonPrj\\bodyEstimation\\static\\output')
            print(result['candidate'])

            path = '/\\static\\output'
            txt_name0 = "out.jpg"
            os.remove(os.path.join(path, txt_name0))
            filelist = os.listdir(path)
            for item in filelist:
                if item.endswith('.jpg'):
                    name = item.split('.', 1)[0]
                    src = os.path.join(os.path.abspath(path), item)
                    dst = os.path.join(os.path.abspath(path), 'out.jpg')
                try:
                    os.rename(src, dst)
                    print('rename from %s to %s' % (src, dst))
                except:
                    continue
            for item in filelist:
                if item.endswith('.png'):
                    name = item.split('.', 1)[0]
                    src = os.path.join(os.path.abspath(path), item)
                    dst = os.path.join(os.path.abspath(path), 'out.jpg')
                try:
                    os.rename(src, dst)
                    print('rename from %s to %s' % (src, dst))
                except:
                    continue
            for item in filelist:
                if item.endswith('.jpeg'):
                    name = item.split('.', 1)[0]
                    src = os.path.join(os.path.abspath(path), item)
                    dst = os.path.join(os.path.abspath(path), 'out.jpg')
                try:
                    os.rename(src, dst)
                    print('rename from %s to %s' % (src, dst))
                except:
                    continue

            # image = cv2.imread('keypoint_body.png')
            # jpeg = turbojpeg.encode(image)
            # resp = make_response(jpeg)
            # resp.headers['Content-Type'] = 'image/jpeg'
            # resp.headers['image-focus'] = image_focus
            return render_template('index.html')
            # return render_template('index.html', myImg = objFile)
        else:
            return "error"
    else:
        return "error"


if __name__ == '__main__':
    app.run()
