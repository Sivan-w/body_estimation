from flask import Flask, make_response, render_template, request
import caculate_angle as ca
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
someadvice = "欢迎使用坐姿检测器(｡･∀･)ﾉﾞ嗨"


@app.route('/index')
def indexes():
    return render_template('index.html', advice=someadvice)


@app.route('/upload', methods=['GET', 'POST'])
def uploadImg():
    if request.method == 'GET':
        return render_template('index.html', advice=someadvice)
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
            p = ca.PoseAnalyzer(result)
            advice = p.logic_realize()
            print(advice)
            print(result['candidate'])
            filelist = os.listdir(path)

            suffix = (".jpg", ".png", ".jpeg")
            for x in suffix:
                sth.reName(x, filelist, path)

            return render_template('index.html', advice=advice)
            # return render_template('index.html', myImg = objFile)
        else:
            return "error"
    else:
        return "error"


if __name__ == '__main__':
    app.run(
        port=80
    )
