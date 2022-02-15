from flask import Flask, make_response, render_template, request
from turbojpeg import TurboJPEG
import paddlehub as hub
import numpy as np
import base64
import requests
import sth
import cv2
import shutil  # 用于清空文件夹。替代os库，os存在限制
import os

app = Flask(__name__)
model = hub.Module(name='openpose_body_estimation')


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
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        if 'myImg' in request.files:
            objFile = request.files.get('myImg')
            objFile.filename = "img.jpg"
            strFilePath = "./static/imgs/" + objFile.filename
            objFile.save(strFilePath)
            path = 'D:\\project\\pythonPrj\\body_estimation\\static\\output'
            # 判断是否存在output文件夹（可能上次运行时被rmtree删了
            if not os.path.exists(path):
                os.mkdir(path)
            # 清空文件夹
            shutil.rmtree(path)
            # os.remove(path)

            # 跑模型
            result = model.predict('D:\\project\\pythonPrj\\body_estimation\\static\\imgs\\' + objFile.filename,
                                   path)
            print(result['candidate'])
            filelist = os.listdir(path)

            suffix = (".jpg", ".png", ".jpeg")
            for x in suffix:
                sth.reName(x, filelist, path)

            # for item in filelist:
            #     if item.endswith('.png'):
            #         name = item.split('.', 1)[0]
            #         src = os.path.join(os.path.abspath(path), item)
            #         dst = os.path.join(os.path.abspath(path), 'out.jpg')
            #     try:
            #         os.rename(src, dst)
            #         print('rename from %s to %s' % (src, dst))
            #     except:
            #         continue

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
