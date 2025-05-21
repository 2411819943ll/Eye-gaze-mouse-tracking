
# For monitoring web camera and performing image minipulations
import cv2
from cv2 import CascadeClassifier, VideoCapture
import numpy as np
import os
import shutil
from pynput.mouse import Listener
from cv2 import *

# root = input("Enter the directory to store the images: ")
# if os.path.isdir(root):
#     resp = ""
# while not resp in ["Y", "N"]:
#     resp = input("This directory already exists. If you continue, the contents of the existing directory will be deleted. If you would still like to proceed, enter [Y]. Otherwise, enter [N]: ")
# if resp == "Y": 
#     shutil.rmtree(root)
# else:
#     exit()
# os.mkdir(root)

# Normalization helper function
def normalize(x):
    minn, maxx = x.min(), x.max()
    return (x - minn) / (maxx - minn)

# Eye cropping function
# 定义一个函数scan，用于扫描图像
    # 读取视频帧
def scan(image_size=(32, 32)):
    # 从视频捕获中读取一帧
    # 将图像转换为灰度图像
    _, frame = video_capture.read()
    # 使用级联分类器检测图像中的目标
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    boxes = cascade.detectMultiScale(gray, 1.3, 10)
    if len(boxes) == 2:
        eyes = []
        for box in boxes:
            x, y, w, h = box
            eye = frame[y:y + h, x:x + w]
            eye = cv2.resize(eye, image_size)
            eye = normalize(eye)
            eye = eye[10:-10, 5:-5]
            eyes.append(eye)
        return (np.hstack(eyes) * 255).astype(np.uint8)
    else:
        return None

def on_click(x, y, button, pressed):
    # If the action was a mouse PRESS (not a RELEASE)
    eyes = None
    if pressed:
    # Crop the eyes
        eyes = scan()
    # If the function returned None, something went wrong
    if not eyes is None:
    # Save the image
        filename = "{} {} {}.jpeg".format(x, y, button)
        cv2.imwrite(filename, eyes)
        print(filename)

cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
video_capture = cv2.VideoCapture(0)

# 设置相机分辨率为1280x720
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 验证设置是否成功
actual_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"相机分辨率设置为: {actual_width}x{actual_height}")

# with Listener(on_click = on_click) as listener:
#     listener.join()